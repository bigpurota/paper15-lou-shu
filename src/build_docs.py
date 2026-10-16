"""Fill every number in README.md and slides/PLAN.md from the CSVs under results/.

The two documents are written by hand, but no number in them is typed by hand. A number sits either
inside an inline marker or inside a generated block:

    <!--v:key-->1.70 (t 1.12)<!--/v-->                  one value, refreshed on every render
    <!-- BEGIN:name --> ... <!-- END:name -->           a generated table or list

A fresh placeholder ``{{key}}`` written into the prose is turned into an inline marker on the next
render. Text outside markers is never touched. Values come from the base run that results/LATEST points
to, from results/simulations/ and from results/upgrades/*/; ``--list`` prints every available key.

Usage:
    python -I src/build_docs.py [--results-dir results] [--check] [--strict] [--list]
        --check   write nothing; exit 1 if README.md or slides/PLAN.md is out of date
        --strict  exit 1 if a decimal number appears outside markers (a hand-typed number)
        --list    print all keys with their values and exit
Writes: README.md, slides/PLAN.md, results/docs_manifest.json (inputs with sha256, git state).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "slides" / "PLAN.md"]
DECK = ROOT / "slides" / "deck.tex"          # hand-written slide text; numbers only via \V{key}
GEN = ROOT / "slides" / "gen"                # generated: values.tex (every key) and the slide tables
HLZ_DEFAULT = 3.0

INLINE_RE = re.compile(r"<!--v:([A-Za-z0-9_.\-]+)-->(.*?)<!--/v-->")
PLACEHOLDER_RE = re.compile(r"\{\{([A-Za-z0-9_.\-]+)\}\}")
BLOCK_RE = re.compile(r"(<!-- BEGIN:([A-Za-z0-9_.\-]+) -->)(.*?)(<!-- END:\2 -->)", re.S)


# ------------------------------------------------------------------------------------ formatting
def fx(x, d: int = 2) -> str:
    """Fixed decimals; refuses NaN so that a missing input fails loudly instead of printing 'nan'."""
    x = float(x)
    if not math.isfinite(x):
        raise ValueError("non-finite value")
    s = f"{x:.{d}f}"
    if s.startswith("-") and float(s) == 0:
        s = s[1:]
    return s


def fa(x) -> str:
    """Coefficient with 3 decimals below 1 in absolute value, 2 otherwise."""
    return fx(x, 3 if abs(float(x)) < 1 else 2)


def fp(x) -> str:
    """Paper number as printed (up to 3 decimals, trailing zeros removed, at least 2 decimals)."""
    s = f"{float(x):.3f}".rstrip("0")
    if len(s.split(".")[1]) < 2:
        s = f"{float(x):.2f}"
    return s


def ct(c, t, d: int | None = None) -> str:
    return f"{fa(c) if d is None else fx(c, d)} (t {fx(t, 2)})"


def pc(share, d: int = 0) -> str:
    return fx(100 * float(share), d) + "%"


def num(n) -> str:
    """Integer with a space as thousands separator."""
    return f"{int(round(float(n))):,}".replace(",", " ")


def fsec(x) -> str:
    """Seconds: one decimal below 10 s, whole seconds above."""
    return fx(x, 1 if float(x) < 10 else 0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------ registry
class Values(dict):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: dict[str, str] = {}

    def put(self, key: str, value) -> None:
        if key in self:
            raise KeyError(f"duplicate key {key}")
        value = str(value)
        if "\n" in value or "<!--" in value:
            raise ValueError(f"inline value for {key} must be one line without markers")
        self[key] = value

    def csv(self, path: Path) -> pd.DataFrame:
        self.inputs[str(path.relative_to(ROOT))] = sha256(path)
        return pd.read_csv(path)

    def json(self, path: Path) -> dict:
        self.inputs[str(path.relative_to(ROOT))] = sha256(path)
        return json.loads(path.read_text())

    def text(self, path: Path) -> str:
        self.inputs[str(path.relative_to(ROOT))] = sha256(path)
        return path.read_text()


def one(df: pd.DataFrame, **eq) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for k, v in eq.items():
        m &= df[k] == v
    out = df[m]
    if len(out) != 1:
        raise LookupError(f"expected 1 row for {eq}, got {len(out)}")
    return out.iloc[0]


def md_table(header: list[str], rows: list[list[str]]) -> str:
    esc = lambda s: str(s).replace("|", "\\|")  # noqa: E731
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"sha": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


# ------------------------------------------------------------------------------------ base run
def base_values(V: Values, B: dict, R: Path) -> None:
    rd = R / (R / "LATEST").read_text().strip()
    B["dir"] = rd
    man = V.json(rd / "run_manifest.json")
    B["manifest"] = man
    V.put("base.run", man["run_id"])
    V.put("base.dir", f"results/{man['run_id']}")
    V.put("base.git", man["git"]["sha"][:7] + (" (dirty)" if man["git"]["dirty"] else ""))
    V.put("base.seed", man["seed"])
    V.put("base.cfg8", man["config_sha256"][:8])
    V.put("base.t_total", fx(man["timings_sec"]["total"], 0))
    V.put("base.rss", fx(man.get("peak_rss_mb", float("nan")), 0))

    s = V.csv(rd / "sample.csv").iloc[0]
    V.put("n.months", int(s.n_months))
    V.put("n.first", s["first"])
    V.put("n.last", s["last"])
    V.put("n.coinmonths", num(s.n_coin_months))
    V.put("n.coins", num(s.n_coins))
    V.put("n.avgcoins", fx(s.avg_coins_per_month, 0))
    t4 = V.csv(rd / "T4_fm.csv")
    V.put("n.fmmonths", int(t4.n_months.iloc[0]))
    V.put("n.fmobs", fx(t4.avg_obs.iloc[0], 0))
    V.put("n.avgcoins_fm_all", fx(s.avg_coins_fm, 0))
    V.put("n.fm_dropped", int(s.n_months) - int(t4.n_months.iloc[0]))
    V.put("d.daily_rows", num(man["rows"]["daily_rows"]))
    V.put("d.daily_symbols", num(man["rows"]["daily_symbols"]))

    # labels already formatted by report.py
    for r in V.csv(rd / "report_diagnostics.csv").itertuples():
        V.put(f"d.{r.id}", r.label)

    # paper vs us rows
    pvu = V.csv(rd / "paper_vs_us.csv")
    B["pvu"] = pvu
    for r in pvu.itertuples():
        if r.kind == "t":
            V.put(f"pvu.{r.id}", ct(r.us, r.us_t))
            V.put(f"pvu.{r.id}.paper", f"{fp(r.paper)} (t {fx(r.paper_t)})")
        else:
            d = 3 if r.kind == "level" else 2
            V.put(f"pvu.{r.id}", fx(r.us, d))
            V.put(f"pvu.{r.id}.paper", fx(r.paper, d))
        V.put(f"pvu.{r.id}.match", r.match)
    ls = pvu[(pvu.kind == "t") & (pvu.section != "A&N")]
    an = pvu[(pvu.kind == "t") & (pvu.section == "A&N")]
    V.put("sc.ls_n", len(ls))
    V.put("sc.yes", int((ls.match == "yes").sum()))
    V.put("sc.sign", int((ls.match == "sign only").sum()))
    V.put("sc.no", int((ls.match == "no").sum()))
    V.put("sc.sig", int((ls.us_t.abs() >= 1.96).sum()))
    V.put("sc.hlz", int((ls.us_t.abs() > HLZ_DEFAULT).sum()))
    V.put("sc.an_n", len(an))
    V.put("sc.an_opp", int((an.match == "no").sum()))
    V.put("sc.samesign", int((ls.us * ls.paper > 0).sum()))
    res = ls[ls.id.str.contains("ResLnA")]
    V.put("sc.res_n", len(res))
    V.put("sc.res_sig", int((res.us_t.abs() >= 1.96).sum()))
    # full-sample rows of the volume part (A and A_C: sorts, alphas, FM slopes); sub-period rows excluded
    vol = ls[ls.id.isin(["HLArawCoef", "HLAalphaCoef", "HLACrawCoef", "HLACalphaCoef", "FMlnA", "FMlnAC",
                         "FMlnACfour", "FMlnACfive", "TtwolnAC", "RawlnA", "RawlnAC"])]
    V.put("sc.vol_n", len(vol))
    V.put("sc.vol_sig", int((vol.us_t.abs() >= 1.96).sum()))
    lv = pvu[pvu.kind != "t"]
    V.put("sc.lvl_n", len(lv))
    V.put("sc.lvl_yes", int((lv.match == "yes").sum()))

    # quintiles, ours and paper
    t3 = V.csv(rd / "T3_sorts.csv")
    pn = yaml.safe_load(V.text(ROOT / "src" / "paper_numbers.yaml"))
    B["paper"] = pn
    for sig in ["A", "A_C", "ResA"]:
        r = one(t3, row="raw", signal=sig)
        V.put(f"q.{sig}", " / ".join(fx(r[f"Q{i}"]) for i in range(1, 6)))
        V.put(f"q.{sig}.paper", " / ".join(fx(x) for x in pn["lou_shu_2017"]["table3_panelA"][sig]["raw"]["q"]))
        V.put(f"q.{sig}.min", fx(min(r[f"Q{i}"] for i in range(1, 6)), 1))
        V.put(f"q.{sig}.max", fx(max(r[f"Q{i}"] for i in range(1, 6)), 1))

    t5 = V.csv(rd / "T5_imla.csv")
    for key, model in [("raw", "raw"), ("imla", "IMLA_C"), ("imla3", "IMLA_C+CMKT+CMOM")]:
        r = one(t5, model=model)
        V.put(f"t5.{key}", ct(r.alpha_HL_A, r.t))

    t7 = V.csv(rd / "T7_january.csv")
    for key, sub in [("jan", "January"), ("rest", "Feb-Dec")]:
        r = one(t7, subset=sub)
        V.put(f"t7.{key}", ct(r.coef, r.t))
        V.put(f"t7.{key}.n", int(r.n))

    rob = V.csv(rd / "robustness_vs_base.csv")
    V.put("rob.n", len(rob))
    V.put("rob.same", int((rob.same_conclusion == "yes").sum()))
    r = one(rob, statistic="HL A_C raw", variant="EW, liquid half")
    V.put("rv.liq.hlAC", ct(r.variant_coef, r.variant_t))

    aud = V.csv(rd / "audit_summary.csv").set_index("type")
    for t in ["redenomination", "ticker_reuse", "long_gap", "zero_volume", "open_mismatch", "TOTAL"]:
        V.put(f"audit.{t}", int(aud.loc[t, "n_events"]) if t in aud.index else 0)
    V.put("audit.rows_in", num(aud.loc["rows_in", "n_events"]))
    V.put("audit.rows_out", num(aud.loc["rows_out", "n_events"]))
    V.put("audit.ids_in", num(aud.loc["rows_in", "n_symbols"]))
    V.put("audit.ids_out", num(aud.loc["rows_out", "n_symbols"]))
    ev = V.csv(rd / "audit_events.csv")
    B["audit_events"] = ev

    rw = V.csv(rd / "R_winsor_returns.csv")
    B["winsor"] = rw

    fac = V.csv(rd / "factors.csv")
    V.put("f.cmkt_mean", fx(fac.CMKT.mean(), 2))

    plc = V.csv(rd / "placebo.csv")
    sh = plc[plc.test == "shuffle within month"]
    V.put("plc.shuffle_reps", int(sh.n.iloc[0]))
    # placebo.py: for the shuffle rows `t` is the mean t over the permutations, `sd_t` its s.d. across them
    V.put("plc.shuffle_maxabs_meant", fx(sh.t.abs().max(), 2))
    V.put("plc.shuffle_sd_t_min", fx(sh.sd_t.min(), 2))
    V.put("plc.shuffle_sd_t_max", fx(sh.sd_t.max(), 2))
    V.put("plc.shuffle_share196_max", pc(sh["share_abs_t_gt_1.96"].max()))
    lk = plc[plc.test.str.startswith("leak")]
    V.put("plc.leak_minabs_t", fx(lk.t.abs().min(), 2))
    V.put("plc.leak_maxabs_t", fx(lk.t.abs().max(), 2))


# ------------------------------------------------------------------------------------ simulations
def sim_values(V: Values, B: dict, R: Path) -> None:
    sd = R / "simulations"
    ws = V.csv(sd / "sim_world_summary.csv")
    B["sim_ws"] = ws
    sm = V.json(sd / "sim_manifest.json")
    V.put("sim.wall", fx(sm["wall_sec"], 0))
    V.put("sim.run", sm["run_id"])
    V.put("sim.cfg8", sm["config_sha256"][:8])
    V.put("sim.git", sm["git_sha"][:7] + (" (dirty)" if sm.get("git_dirty") else ""))
    V.put("sim.assets", sm["n_assets"])
    V.put("sim.months", sm["n_months"])
    for w in ["V", "P", "0"]:
        V.put(f"sim.{w}.reps", int(ws[ws.world == w].n_reps.iloc[0]))

    def g(w, stat, col):
        return one(ws, world=w, stat=stat)[col]
    V.put("sim.V.fm2", pc(g("V", "fm2_lnA_C", "share_t_gt_1.96")))
    V.put("sim.V.fm2.t", fx(g("V", "fm2_lnA_C", "mean_t"), 1))
    V.put("sim.V.hlAC", pc(g("V", "hl_A_C", "share_t_gt_1.96")))
    V.put("sim.V.fm3", pc(g("V", "fm3_ResLnA", "share_abs_t_gt_1.96")))
    V.put("sim.V.hlResA", pc(g("V", "hl_ResA", "share_abs_t_gt_1.96")))
    V.put("sim.P.fm3", pc(g("P", "fm3_ResLnA", "share_t_gt_1.96")))
    V.put("sim.P.fm3.t", fx(g("P", "fm3_ResLnA", "mean_t"), 1))
    V.put("sim.P.hlResA", pc(g("P", "hl_ResA", "share_t_gt_1.96")))
    V.put("sim.P.fm2", pc(g("P", "fm2_lnA_C", "share_abs_t_gt_1.96")))
    for stat, key in [("fm2_lnA_C", "fm2"), ("fm3_ResLnA", "fm3"), ("hl_A_C", "hlAC"), ("hl_A", "hlA"),
                      ("hl_ResA", "hlResA"), ("fm3_ResLnA_nw0", "fm3nw0"), ("fm2_lnA_C_nw0", "fm2nw0")]:
        V.put(f"sim.0.{key}", pc(g("0", stat, "share_abs_t_gt_1.96"), 1))
        V.put(f"sim.0.{key}.3", pc(g("0", stat, "share_abs_t_gt_3"), 1))
    chk = V.csv(sd / "sim_checks.csv")
    passed = chk["pass"].astype(str).str.lower() == "true"
    V.put("sim.checks.pass", int(passed.sum()))
    V.put("sim.checks.n", len(chk))
    B["sim_checks"] = chk
    fails = []
    for r in chk[~passed].itertuples():
        stat = re.search(r"t\(([^)]*)\)", r.criterion).group(1)
        ci = re.search(r"CI ([0-9.]+), ([0-9.]+)", r.criterion)
        ge = re.match(r">=\s*([0-9.]+)", str(r.rule))
        if ge:
            why = f"при правиле ≥ {pc(float(ge.group(1)))}"
        elif ci:
            why = f"95% ДИ {pc(float(ci.group(1)), 1)}–{pc(float(ci.group(2)), 1)} не содержит 5%"
        else:
            why = f"правило {r.rule}"
        fails.append(f"мир {r.world}, {stat}: {pc(r.value, 1)} ({why})")
    V.put("sim.checks.failed", "; ".join(fails) if fails else "нет")
    pw = V.csv(sd / "sim_power.csv")
    B["sim_power"] = pw
    for prem in sorted(pw.premium.unique()):
        for stat, key in [("fm2_lnA_C", "fm2"), ("hl_A_C", "hlAC")]:
            r = one(pw, premium=prem, stat=stat)
            V.put(f"pow.{fx(100 * prem, 2)}.{key}", pc(r["share_t_gt_1.96"]))
    V.put("pow.reps", int(pw.n_reps.min()))


# ------------------------------------------------------------------------------------ upgrades
def manifest_id(V: Values, B: dict, U: Path, name: str) -> dict:
    m = V.json(U / name / "run_manifest.json")
    V.put(f"run.{name}", m["run_id"])
    B.setdefault("upgrade_manifests", {})[name] = m
    return m


def an_values(V: Values, B: dict, U: Path) -> None:
    d = U / "amihud_noh"
    manifest_id(V, B, U, "amihud_noh")
    ds = V.csv(d / "describe.csv").set_index("sample")
    V.put("an.covneg", pc(ds.loc["monthly", "share_COV_negative_pooled"], 1))
    V.put("an.covneg12", pc(ds.loc["12m", "share_COV_negative_pooled"], 1))
    V.put("an.mean_dif", ct(ds.loc["monthly", "mean_DIF"], ds.loc["monthly", "t_mean_DIF"]))
    V.put("an.mean_dif12", ct(ds.loc["12m", "mean_DIF"], ds.loc["12m", "t_mean_DIF"]))
    V.put("an.sd_dif12", fx(ds.loc["12m", "sd_DIF_avg_month"], 3))
    V.put("an.r2_dif", fx(ds.loc["12m", "r2_DIF_on_components_int"], 2))
    V.put("an.sd_retadj", fx(ds.loc["12m", "cs_sd_ret_adj_avg_month"], 1))
    pr = V.csv(d / "paper_reference.csv")
    V.put("an.p.mean_dif", fp(one(pr, model="AN descriptive", var="mean_DIF").coef))
    V.put("an.p.sd_dif", fp(one(pr, model="AN descriptive", var="sd_DIF").coef))
    for key, model, var in [("col2", "AN col2", "DIF"), ("col3", "AN col3", "DIF"), ("col4", "AN col4", "RlnILLIQ"),
                            ("fdif", "AN fDIF_int", "fDIF_int")]:
        r = one(pr, model=model, var=var)
        V.put(f"an.p.{key}", f"{fp(r.coef)} (t {fx(r.t)})")
    vd = V.csv(d / "verdict.csv")
    for test in ["V1", "V2", "V3", "V4", "V5", "V6"]:
        for sample, sk in [("monthly", "m"), ("12m", "y")]:
            r = vd[(vd.test == test) & (vd["sample"] == sample) & (vd.dep == "ret_adj")]
            if len(r) == 1:
                r = r.iloc[0]
                V.put(f"an.{test}{sk}", ct(r.coef, r.t))
                V.put(f"an.{test}{sk}.var", r["var"])
    vc = V.csv(d / "verdict_counts.csv").set_index("dep")
    for dep in ["ret_adj", "ret"]:
        V.put(f"an.verdict.{dep}", vc.loc[dep, "verdict"])
        V.put(f"an.n.{dep}", int(vc.loc[dep, "n_tests"]))
        for c in ["AN_1.96", "LS_1.96", "LS_null_1.96", "none_1.96", "AN_3", "LS_3"]:
            V.put(f"an.cnt.{dep}.{c}", int(vc.loc[dep, c]))
    sc = V.csv(d / "sensitivity_min_obs_counts.csv")
    order = list(dict.fromkeys(sc.min_obs))
    V.put("an.sens.levels", " / ".join(order))
    for dep in ["ret_adj", "ret"]:
        V.put(f"an.sens.{dep}", " / ".join(one(sc, dep=dep, min_obs=o).verdict for o in order))
    so = V.csv(d / "sensitivity_min_obs.csv")
    V.put("an.sens.maxabs_t", fx(so.t.abs().max(), 2))
    V.put("an.sens.n3", int((so.t.abs() >= 3).sum()))
    V.put("an.minN", int(so.min_n.min()))
    iv = V.csv(d / "ivol_channel.csv").set_index("sample")
    V.put("an.corr_res_absret", fx(iv.loc["monthly", "corr_res_ln_absret"], 2))
    V.put("an.r2_res", fx(iv.loc["monthly", "r2_res_on_ln_absret_ivol"], 2))
    V.put("an.corr_res_dif", fx(iv.loc["monthly", "corr_res_perp_DIF_perp"], 2))
    V.put("an.r2_res_abs", fx(iv.loc["monthly", "r2_res_on_ln_absret"], 2))
    V.put("an.r2_res_dif", fx(iv.loc["monthly", "r2_res_on_ln_absret_DIF"], 2))
    fm = V.csv(d / "fm.csv")
    fm = fm[(fm["sample"] == "monthly") & (fm.dep == "ret_adj") & (fm["var"] == "ResLnA")]
    for key, pref in [("c1", "C1"), ("c4", "C4"), ("c5", "C5")]:
        r = fm[fm.model.str.startswith(pref)]
        if len(r) == 1:
            V.put(f"an.{key}", ct(r.iloc[0].coef, r.iloc[0].t))


def hf_values(V: Values, B: dict, U: Path) -> None:
    d = U / "hf_lambda"
    manifest_id(V, B, U, "hf_lambda")
    cov = V.csv(d / "coverage.csv").iloc[0]
    V.put("hf.months", int(cov.n_measure_months))
    V.put("hf.coinmonths", num(cov.n_lambda_ok))
    # coverage.csv stores GiB (bytes / 2**30); the README data table uses decimal MB, so convert to 10^9 bytes
    V.put("hf.gb5m", fx(cov.klines5m_gb * 2**30 / 1e9, 1))
    V.put("hf.gbagg", fx(cov.aggtrades_gb * 2**30 / 1e9, 1))
    sl = V.csv(d / "fm_slopes_monthly.csv")
    V.put("hf.first", sl.month.min())
    V.put("hf.last", sl.month.max())
    a = V.csv(d / "T9A_corr.csv")
    for v in ["A", "A_C", "ResA"]:
        r = one(a, spec="main", corr="levels_pearson", var=v)
        V.put(f"hf.corr.{v}", fx(r.ours, 3))
        V.put(f"hf.corr.{v}.paper", fx(r.paper, 3))
    b = V.csv(d / "T9B_fm.csv")
    b = b[b.spec == "main"]
    V.put("hf.obs", fx(b.avg_obs.iloc[0], 0))
    for key, model, var in [("b0", "(0) same sample", "lnA_C"), ("b1", "(1)", "lnlam"), ("b2lam", "(2)", "lnlam"),
                            ("b2ac", "(2)", "lnA_C")]:
        r = one(b, model=model, var=var)
        V.put(f"hf.{key}", ct(r.coef, r.t))
        if pd.notna(r.paper_coef):
            V.put(f"hf.{key}.paper", f"{fp(r.paper_coef)} (t {fx(r.paper_t)})")
    t10 = V.csv(d / "T10_fm.csv")
    t10 = t10[(t10.spec == "main") & (t10.cost_benchmarks == "lambda")]
    V.put("hf.t10r2", fx(t10.avg_cs_r2_lnA_on_cost.iloc[0], 2))
    for key, model, var in [("c1", "(1)", "cost"), ("n2", "(2)", "noncost"), ("c3", "(3)", "cost"),
                            ("n3", "(3)", "noncost")]:
        r = one(t10, model=model, var=var)
        V.put(f"hf.t10.{key}", ct(r.coef, r.t))
        V.put(f"hf.t10.{key}.paper", f"{fp(r.paper_coef)} (t {fx(r.paper_t)})")
    dg = V.csv(d / "T9B_diagnostics.csv")
    vol = dg[dg.variant.str.startswith("+ lnVOL")]
    V.put("hf.vol.b1", ct(one(vol, model="(1)", var="lnlam").coef, one(vol, model="(1)", var="lnlam").t))
    r = one(vol, model="(2)", var="lnA_C")
    V.put("hf.vol.b2ac", ct(r.coef, r.t))
    for key, pref in [("h1", "first half"), ("h2", "second half")]:
        r = one(dg[dg.variant.str.startswith(pref)], model="(2)", var="lnA_C")
        V.put(f"hf.{key}.ac_t", fx(r.t, 2))
    lam = dg[(dg["var"] == "lnlam")]
    V.put("hf.lam_tmin", fx(lam.t.abs().min(), 2))
    V.put("hf.lam_tmax", fx(lam.t.abs().max(), 2))
    vs = V.csv(d / "validation_summary.csv")
    r = one(vs, pair="exact_vs_kl_sqrt_sides", months="all")
    V.put("hf.val.corr", fx(r.pearson_log, 2))
    V.put("hf.val.n", int(r.n))


def edge_values(V: Values, B: dict, U: Path) -> None:
    d = U / "edge_spread"
    m = manifest_id(V, B, U, "edge_spread")
    ed = V.csv(d / "edge_describe.csv").iloc[0]
    V.put("ed.n", num(ed.n_coin_months))
    V.put("ed.nonpos", pc(ed.share_nonpos, 1))
    V.put("ed.w2", fx(ed.median_weight_moment2, 4))
    V.put("ed.gap_bp", fx(ed.median_abs_open_gap_bp, 1))
    V.put("ed.gap_zero", pc(ed.share_open_eq_prev_close))
    V.put("ed.bidask_diff", fx(m["checks"]["bidask_max_abs_diff"], 1))
    V.put("ed.bidask_n", m["checks"]["bidask_n_coin_months"])
    c = V.csv(d / "T9C_corr.csv")
    r = one(c, method="pearson", window="m1", rule="floor")
    pn = V.csv(d / "paper_numbers.csv").set_index("key")["value"]
    for v, pk in [("A", "corr_ES_A"), ("A_C", "corr_ES_A_C"), ("ResA", "corr_ES_ResA")]:
        V.put(f"ed.corr.{v}", fx(r[f"corr_EDGE_{v}"], 3))
        V.put(f"ed.corr.{v}.paper", fx(pn[pk], 3))
    f = V.csv(d / "T9C_fm.csv")
    main = f[(f.window == "m1") & (f.rule == "floor")]
    r = one(main, model="(1) lnEDGE", var="lnEDGE")
    V.put("ed.fm1", ct(r.coef, r.t))
    V.put("ed.fm1.paper", f"{fp(pn['fm_lnES_alone.coef'])} (t {fx(pn['fm_lnES_alone.t'])})")
    r = one(main, model="(2) lnEDGE+lnA_C", var="lnA_C")
    V.put("ed.fm2ac", ct(r.coef, r.t))
    alone = f[(f.model == "(1) lnEDGE") & (f["var"] == "lnEDGE")]
    V.put("ed.alone_pos", f"{int((alone.coef > 0).sum())} of {len(alone)}")
    t10 = V.csv(d / "T10_fm.csv")
    r = one(t10[(t10.window == "m1") & (t10.rule == "floor")], model="(1) cost", var="Cost")
    V.put("ed.t10cost", ct(r.coef, r.t))
    r = one(t10[(t10.window == "m1") & (t10.rule == "floor")], model="(2) non-cost", var="NonCost")
    V.put("ed.t10non", ct(r.coef, r.t))
    cost = t10[(t10.model == "(1) cost") & (t10["var"] == "Cost")]
    V.put("ed.cost_pos", f"{int((cost.coef > 0).sum())} of {len(cost)}")
    allt = pd.concat([f[f["var"].isin(["lnEDGE", "lnA_C"]) & ~f.model.str.startswith("(ref)")].t,
                      t10[t10["var"].isin(["Cost", "NonCost"])].t])
    V.put("ed.maxabs_t", fx(allt.abs().max(), 2))
    cs = V.csv(d / "costs_summary.csv")
    r = one(cs, signal="A_C", rule="pooled")
    V.put("ed.months", int(r.n_months))
    V.put("ed.gross", ct(r.hl_gross_pct, r.hl_gross_t))
    V.put("ed.net", ct(r.hl_net_pct, r.hl_net_t))
    V.put("ed.netfee", ct(r.hl_net_fee_pct, r.hl_net_fee_t))
    V.put("ed.fee", fx(r.fee_per_side_pct, 1) + "%")
    V.put("ed.breakeven", fx(r.breakeven_cost_per_side_pct, 2) + "%")
    V.put("ed.hs0_long", pc(r.share_months_hs_zero_long))
    V.put("ed.hs0_short", pc(r.share_months_hs_zero_short))
    V.put("ed.turn_long", fx(r.turnover_long, 2))
    V.put("ed.turn_short", fx(r.turnover_short, 2))


def sentiment_values(V: Values, B: dict, U: Path) -> None:
    d = U / "sentiment"
    manifest_id(V, B, U, "sentiment")
    pv = V.csv(d / "paper_vs_us.csv")
    for test, tk in [("sentiment", "se"), ("illiquidity", "il")]:
        for row, rk in [("Short", "short"), ("Long", "long"), ("Long-Short", "ls"),
                        ("L-S High-Low (Panel A)", "split")]:
            r = one(pv, test=test, row=row)
            V.put(f"{tk}.{rk}", ct(r.ours_b, r.ours_t))
            V.put(f"{tk}.{rk}.paper", f"{fp(r.paper_b)} (t {fx(r.paper_t)})")
        V.put(f"{tk}.n", int(pv[pv.test == test].n_months.iloc[0]))
    pb = V.csv(d / "panel_b.csv")
    groups = {"se": ["sent_mean", "sent_eom"], "il": ["illiq_ar2", "illiq_ar2_log", "illiq_level_log"]}
    main_state = {"se": "sent_mean", "il": "illiq_ar2"}
    for tk, states in groups.items():
        g = pb[pb.state.isin(states) & (pb.portfolio == "Long-Short") & ~pb.post_hoc.astype(bool)]
        V.put(f"{tk}.var_n", len(g))
        V.put(f"{tk}.var_tmin", fx(g.t.min(), 2))
        V.put(f"{tk}.var_tmax", fx(g.t.max(), 2))
        V.put(f"{tk}.var_196", int((g.t.abs() > 1.96).sum()))
        V.put(f"{tk}.var_3", int((g.t.abs() > 3).sum()))
        ph = pb[(pb.state == main_state[tk]) & pb.post_hoc.astype(bool)]
        for port, pk in [("Long-Short", "ls"), ("Short", "short"), ("Long", "long")]:
            r = one(ph, portfolio=port)
            V.put(f"{tk}.post.{pk}", ct(r.b, r.t))
        r = one(pb[(pb.state == main_state[tk]) & pb.main.astype(bool)], portfolio="FM_lnA_C")
        V.put(f"{tk}.fm", ct(r.b, r.t))


def openap_values(V: Values, B: dict, U: Path) -> None:
    d = U / "openap_us"
    manifest_id(V, B, U, "openap_us")
    mn = V.csv(d / "means.csv")
    for sig, sk in [("Illiquidity", "ill"), ("DolVol", "dv")]:
        for per, pk in [("LS_1964_2012", "pre"), ("post_2013", "post"), ("post_2013_minus_LS_1964_2012", "decay")]:
            r = one(mn, signal=sig, period=per)
            V.put(f"op.{sk}.{pk}", ct(r["mean"], r.t))
    r = one(mn, signal="Illiquidity", period="post_2013")
    V.put("op.post_range", f"{r.start}..{r.end}")
    sp = V.csv(d / "spanning.csv")
    for per, pk in [("LS_1964_2012", "pre"), ("post_2013", "post")]:
        for spec, kk in [("one_factor", "one"), ("five_factor", "five"), ("ff4_only", "ff4")]:
            r = one(sp, dep="Illiquidity", period=per, spec=spec)
            V.put(f"op.{pk}.{kk}", ct(r.alpha, r.t_alpha))
    pc_ = V.csv(d / "paper_comparison.csv")
    for item, k in [("TableA1_one_factor_alpha", "one"), ("TableA1_five_factor_alpha", "five"),
                    ("Table3_A_raw_HL", "raw"), ("Table3_A_4factor_alpha_HL", "ff4")]:
        r = one(pc_, item=item)
        V.put(f"op.paper.{k}", f"{fp(r.paper)} (t {fx(r.paper_t)})")
    co = V.csv(d / "correlation.csv")
    for per, k in [("full", "full"), ("LS_1964_2012", "pre"), ("post_2013", "post")]:
        V.put(f"op.corr.{k}", fx(one(co, period=per).pearson, 2))
    ns = V.csv(d / "decay_nw_sensitivity.csv")
    for sig, sk in [("Illiquidity", "ill"), ("DolVol", "dv")]:
        g = ns[ns.signal == sig].sort_values("nw_lags")
        V.put(f"op.{sk}.decay_lags", " / ".join(str(int(x)) for x in g.nw_lags))
        V.put(f"op.{sk}.decay_ts", " / ".join(fx(x, 2) for x in g.t))


def moex_values(V: Values, B: dict, U: Path) -> None:
    d = U / "moex"
    manifest_id(V, B, U, "moex")
    s = one(V.csv(d / "sample.csv"), variant="main")
    V.put("mx.months", int(s.n_months))
    V.put("mx.stocks", int(s.n_stocks))
    V.put("mx.avg", fx(s.avg_stocks_per_month, 0))
    V.put("mx.fmmonths", int(s.n_months_fm_A1))
    V.put("mx.first", s.return_start)
    V.put("mx.last", s.return_end)
    pv = V.csv(d / "paper_vs_moex.csv")
    for item, k in [("corr(A, A_C)", "corrAAC"), ("corr(AT, AT_C)", "corrAT"), ("corr(A, |Ret|)", "corrAabs")]:
        r = one(pv, item=item)
        V.put(f"mx.{k}", fx(r.moex, 3))
        V.put(f"mx.{k}.paper", fx(r.paper, 3))
    for item, k in [("Quintile H-L on A, raw (%/mo)", "hlA"), ("Quintile H-L on A_C, raw (%/mo)", "hlAC"),
                    ("FM (1) lnA", "fm1"), ("FM (2) lnA_C", "fm2"), ("FM (4) lnA_C", "fm4ac"),
                    ("FM (4) ResLnA", "fm4res"), ("FM (5) lnA_C", "fm5ac")]:
        r = one(pv, item=item)
        V.put(f"mx.{k}", ct(r.moex, r.moex_t))
        V.put(f"mx.{k}.paper", f"{fp(r.paper)} (t {fx(r.paper_t)})")
    hl = pv[pv["item"].str.startswith("Quintile H-L on A_C, alpha")].iloc[0]
    V.put("mx.hlACalpha", ct(hl.moex, hl.moex_t))
    hl3 = pv[pv.moex_sig_hlz3.astype(str).str.lower() == "true"]
    V.put("mx.hlz_list", "; ".join(f"{r.item} t {fx(r.moex_t)}" for r in hl3.itertuples()))
    V.put("mx.hlz_n", len(hl3))
    t4 = V.csv(d / "T4_fm.csv")
    t4m = t4[(t4.variant == "main") & t4["var"].isin(["lnA", "lnA_C", "lnAT", "lnAT_C"])]
    V.put("mx.vol_n", len(t4m))
    V.put("mx.vol_196", int((t4m.t > 1.96).sum()))
    ad = one(V.csv(d / "T3_alpha_detail.csv"), signal="A_C")
    V.put("mx.smb", ct(ad.load_CSMB, ad.load_CSMB_t))
    V.put("mx.lnme_q1", fx(ad.lnME_bottom_q, 2))
    V.put("mx.lnme_q5", fx(ad.lnME_top_q, 2))
    V.put("mx.alpha_months", int(ad.n_months_alpha))
    ada = one(V.csv(d / "T3_alpha_detail.csv"), signal="A")
    V.put("mx.hlA_same", ct(ada.raw_hl_same_months, ada.raw_hl_same_months_t))
    by = V.csv(d / "diag_by_year.csv")
    top = by.loc[by.hl_A_C_share_of_cum.idxmax()]
    V.put("mx.topyear", int(top.year))
    V.put("mx.topyear_share", pc(top.hl_A_C_share_of_cum))
    V.put("mx.ex_topyear", ct(top.hl_A_C_excl_year, top.hl_A_C_excl_year_t))
    V.put("mx.fm4_ex_topyear", ct(top.fm4_lnA_C_excl_year, top.fm4_lnA_C_excl_year_t))
    V.put("mx.fm4_loyo_tmin", fx(by[by.n_months_fm > 0].fm4_lnA_C_excl_year_t.min(), 2))


def robustness_values(V: Values, B: dict, U: Path) -> None:
    d = U / "robustness"
    manifest_id(V, B, U, "robustness")
    bc = V.csv(d / "bootstrap_ci.csv").set_index("stat")
    B["boot"] = bc
    V.put("rb.n", len(bc))
    V.put("rb.excl", int(bc.ci_excludes_0.astype(bool).sum()))
    V.put("rb.robust", int((bc.share_seeds_ci_excludes_0 == 1.0).sum()))
    V.put("rb.seeds", int(bc.n_seed_reps.iloc[0]))
    V.put("rb.block", int(bc.block.iloc[0]))
    V.put("rb.draws", int(bc.n_draws.iloc[0]))
    for st in bc.index:
        r = bc.loc[st]
        V.put(f"rb.ci.{st}", f"[{fx(r.ci95_lo)}; {fx(r.ci95_hi)}]")
        V.put(f"rb.seedshare.{st}", f"{int(round(r.share_seeds_ci_excludes_0 * r.n_seed_reps))}/{int(r.n_seed_reps)}")
    lg = V.csv(d / "ledger_summary.csv")
    for src, k in [("BASE", "base"), ("ALL", "all")]:
        r = one(lg, source=src)
        V.put(f"lg.{k}.n", int(r.n_unique))
        V.put(f"lg.{k}.196", int(r["n_abs_t_gt_1.96"]))
        V.put(f"lg.{k}.3", int(r.n_abs_t_gt_3))
        V.put(f"lg.{k}.exp", fx(r["expected_false_1.96_if_all_null"], 1))
        V.put(f"lg.{k}.bonf", int(r.n_bonferroni_sig))
        V.put(f"lg.{k}.bh", int(r.n_bh_sig))
        V.put(f"lg.{k}.crit", fx(r.bonferroni_crit_t, 2))
    tests = lg[(lg.role == "test") & lg.source.str.startswith("upgrade:")]
    V.put("lg.all.3_by_source", ", ".join(f"{r.source.split(':', 1)[1]} {int(r.n_abs_t_gt_3)}"
                                         for r in tests.itertuples() if r.n_abs_t_gt_3 > 0))
    w0 = V.csv(d / "world0_calibration.csv")
    V.put("w0.fpr_min", pc(w0["fpr_1.96"].min(), 1))
    V.put("w0.fpr_max", pc(w0["fpr_1.96"].max(), 1))
    V.put("w0.fpr3_max", pc(w0.fpr_3.max(), 1))
    V.put("w0.crit_min", fx(w0.crit_abs_t_5pct.min(), 2))
    V.put("w0.crit_max", fx(w0.crit_abs_t_5pct.max(), 2))
    V.put("w0.exp_base", fx(w0["fpr_1.96"].mean() * int(one(lg, source="BASE").n_unique), 1))
    mde = V.csv(d / "mde.csv")
    for st, k in [("fm2_lnA_C", "fm2"), ("hl_A_C", "hlAC"), ("hl_A", "hlA")]:
        real = one(mde, stat=st, method="real_se_nw_crit_1.96").mde
        paper = mde[(mde.stat == st) & (mde.world == "paper")].mde.iloc[0]
        V.put(f"mde.{k}", fx(real, 2))
        V.put(f"mde.{k}.paper", fx(paper, 2))
        V.put(f"mde.{k}.ratio", fx(real / paper, 1))
    V.put("mde.fm2.sim", fx(one(mde, stat="fm2_lnA_C", method="sim_scaled_to_real_crit_1.96").mde, 2))
    # per-s.d. numbers are each in their own s.d. (ours: mean monthly cs s.d. in the FM sample; paper: 2.53, p. 13),
    # so also give them per log unit of ln A_C and with the paper's slope scaled to our s.d.
    r_real = one(mde, stat="fm2_lnA_C", method="real_se_nw_crit_1.96")
    r_pap = one(mde, stat="fm2_lnA_C", world="paper")
    V.put("mde.fm2.sd", fx(r_real.sd_signal, 2))
    V.put("mde.fm2.paper_sd", fx(r_pap.sd_signal, 2))
    V.put("mde.fm2.slope", fx(r_real.mde / r_real.sd_signal, 2))
    V.put("mde.fm2.paper_slope", fp(r_pap.mde / r_pap.sd_signal))
    V.put("mde.fm2.slope_ratio", fx((r_real.mde / r_real.sd_signal) / (r_pap.mde / r_pap.sd_signal), 1))
    V.put("mde.fm2.paper_oursd", fx(r_pap.mde / r_pap.sd_signal * r_real.sd_signal, 2))
    ex = V.csv(d / "stability_excl2021.csv")
    for st, k in [("hl_A_C", "hlAC"), ("fm2_lnA_C", "fm2"), ("fm3_ResLnA", "fm3")]:
        r = one(ex, sample="excl_2021", stat=st)
        V.put(f"x21.{k}", ct(r.estimate, r.t_nw))
    lo = V.csv(d / "stability_loyo_summary.csv").set_index("stat")
    for st, k in [("hl_A_C", "hlAC"), ("fm2_lnA_C", "fm2")]:
        V.put(f"loyo.{k}", f"{fx(lo.loc[st, 't_min'])}..{fx(lo.loc[st, 't_max'])}")
        V.put(f"loyo.{k}.n196", int(lo.loc[st, "n_abs_t_gt_1.96"]))
        V.put(f"loyo.{k}.years", int(lo.loc[st, "n_years"]))


# ------------------------------------------------------------------------------------ pre-registration
def prereg_values(V: Values, B: dict, R: Path) -> None:
    """First real-data run (earliest base run manifest) against the pre-registration commit."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout
    mans = []
    for mp in R.glob("*/run_manifest.json"):
        m = json.loads(mp.read_text())
        if "created_utc" in m and "git" in m:
            mans.append((m["created_utc"], mp, m))
    if not mans or B["prereg"] == "n/a":
        for k in ["git.refine", "git.firstrun", "git.firstrun.sha", "git.firstrun.cfg8", "git.prereg.cfgmatch"]:
            V.put(k, "n/a")
        return
    _, mp, first = min(mans, key=lambda x: x[0])
    V.json(mp)
    fsha = first["git"]["sha"]
    refine = git("log", "--format=%h", f"{B['prereg']}..{fsha}", "--", "DEVIATIONS.md", "config.yaml").split()
    V.put("git.refine", ", ".join(f"`{c}`" for c in reversed(refine)).strip("`") if refine else "нет")
    V.put("git.firstrun", first["run_id"])
    V.put("git.firstrun.sha", fsha[:7] + (" (dirty)" if first["git"].get("dirty") else ""))
    V.put("git.firstrun.cfg8", first["config_sha256"][:8])
    pre_cfg = hashlib.sha256(subprocess.run(["git", "show", f"{B['prereg']}:config.yaml"], cwd=ROOT,
                                            capture_output=True).stdout).hexdigest()
    V.put("git.prereg.cfgmatch", "совпадает" if pre_cfg == first["config_sha256"] else "НЕ совпадает")


# ------------------------------------------------------------------------------------ blocks
def blocks(V: Values, B: dict, R: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    rd: Path = B["dir"]
    up = B["upgrade_manifests"]

    out["provenance"] = (
        f"_Числа в этом файле сгенерированы `src/build_docs.py` из CSV: базовый прогон `{V['base.run']}` "
        f"(git `{V['base.git']}`, config sha256 `{V['base.cfg8']}…`), симуляции `results/simulations/` "
        f"(`{V['sim.run']}`, config `{V['sim.cfg8']}…`, git `{V['sim.git']}`), апгрейды "
        + ", ".join(f"`{k}` → `{m['run_id']}`" for k, m in sorted(up.items()))
        + ". Обновление: `make docs`. Числа между маркерами `<!--v:...-->` руками не правятся. "
        "Пока в манифестах стоит `(dirty)`, код, давший результаты, однозначно задают поля `code_sha256` "
        "манифестов (sha256 файлов), а не git SHA._")

    # paper vs us: the generated report itself (sections Key findings + Paper vs us)
    md = V.text(rd / "paper_vs_us.md")
    m = re.search(r"## Key findings\n(.*?)\n## Robustness", md, re.S)
    if not m:
        raise LookupError("paper_vs_us.md: sections not found")
    body = m.group(1).replace("\n## Paper vs us\n", "\n#### Paper vs us\n")
    extra = md_table(
        ["Metric", "Paper", "Us", "Market and window", "Source"],
        [["corr(AT, AT_C)", V["mx.corrAT.paper"], V["mx.corrAT"], f"MOEX TQBR {V['mx.first']}…{V['mx.last']}",
          "`results/upgrades/moex/paper_vs_moex.csv`"],
         ["corr(A, λ)", V["hf.corr.A.paper"], V["hf.corr.A"], f"Binance {V['hf.first']}…{V['hf.last']}; paper T9 1983–2012",
          "`results/upgrades/hf_lambda/T9A_corr.csv`"],
         ["corr(A_C, λ)", V["hf.corr.A_C.paper"], V["hf.corr.A_C"], f"Binance {V['hf.first']}…{V['hf.last']}; paper T9 1983–2012",
          "`results/upgrades/hf_lambda/T9A_corr.csv`"],
         ["FM ln λ, T9B model (1)", V["hf.b1.paper"], V["hf.b1"], f"Binance {V['hf.first']}…{V['hf.last']}; paper T9 1983–2012",
          "`results/upgrades/hf_lambda/T9B_fm.csv`"],
         ["FM ln λ, T9B model (2) with ln A_C", V["hf.b2lam.paper"], V["hf.b2lam"],
          f"Binance {V['hf.first']}…{V['hf.last']}; paper T9 1983–2012", "`results/upgrades/hf_lambda/T9B_fm.csv`"]])
    out["paper_vs_us"] = (f"_Ниже — дословно `{V['base.dir']}/paper_vs_us.md` (генерирует `src/report.py`), "
                          f"разделы «Key findings» и «Paper vs us»._\n\n#### Key findings\n" + body.strip()
                          + "\n\n#### From the upgrades (rows of task §5 that the base run cannot produce)\n\n"
                          "Each row has its own market and window; t in brackets (Newey–West, 6 lags). λ is our "
                          "5-minute bar λ with the trade sign from taker volume, the paper's is TAQ λ with Lee–Ready.\n\n"
                          + extra)

    out["figure_main"] = (f"![Квинтили A / A_C / Res.A: статья и мы]({V['base.dir']}/fig_quintiles.png)\n\n"
                          f"_Файл: `{V['base.dir']}/fig_quintiles.png` (генерирует `src/figures.py`)._")

    # audit list
    ev = B["audit_events"]
    rows = []
    for r in ev.itertuples():
        f = "" if pd.isna(r.factor) else f"{float(r.factor):.4g}"
        rows.append([r.symbol.replace("USDT", ""), r.date, r.type, f, r.action])
    out["audit"] = md_table(["монета", "дата", "тип", "фактор (редоминация) / reopen÷prev close", "обработка"],
                            rows)

    # simulations
    ws = B["sim_ws"]

    def srow(w, stat, col, expect):
        r = one(ws, world=w, stat=stat)
        return [w, stat, expect, pc(r[col], 1), fx(r.mean_t, 2), int(r.n_reps)]
    srows = [
        srow("V", "fm2_lnA_C", "share_t_gt_1.96", "прайсится: доля t>1.96 высокая"),
        srow("V", "hl_A_C", "share_t_gt_1.96", "прайсится"),
        srow("V", "fm3_ResLnA", "share_abs_t_gt_1.96", "не прайсится: доля |t|>1.96 около 5%"),
        srow("V", "hl_ResA", "share_abs_t_gt_1.96", "не прайсится"),
        srow("P", "fm3_ResLnA", "share_t_gt_1.96", "прайсится"),
        srow("P", "hl_ResA", "share_t_gt_1.96", "прайсится (сортировка слабее FM)"),
        srow("P", "fm2_lnA_C", "share_abs_t_gt_1.96", "в идеале не прайсится"),
        srow("0", "fm2_lnA_C", "share_abs_t_gt_1.96", "ложные значимости около 5%"),
        srow("0", "fm3_ResLnA", "share_abs_t_gt_1.96", "ложные значимости около 5%"),
        srow("0", "fm3_ResLnA_nw0", "share_abs_t_gt_1.96", "то же без NW-лагов"),
        srow("0", "hl_A_C", "share_abs_t_gt_1.96", "ложные значимости около 5%"),
    ]
    t = md_table(["мир", "статистика", "ожидание", "доля значимых", "средний t", "повторов"], srows)
    B.setdefault("tables", {})["sims"] = (["мир", "статистика", "ожидание", "доля значимых", "средний t", "повторов"],
                                          [[str(c) for c in r] for r in srows])
    chk = B["sim_checks"]
    failed = chk[chk["pass"].astype(str).str.lower() != "true"]
    fl = "\n".join(f"- не пройдено: мир {r.world}, {r.criterion} = {fx(r.value, 3)} (правило `{r.rule}`)"
                   for r in failed.itertuples())
    pw = B["sim_power"]
    prow = []
    for prem in sorted(pw.premium.unique()):
        a = one(pw, premium=prem, stat="fm2_lnA_C")
        b = one(pw, premium=prem, stat="hl_A_C")
        prow.append([fx(100 * prem, 2), pc(a["share_t_gt_1.96"]), pc(b["share_t_gt_1.96"]), int(a.n_reps)])
    pt = md_table(["премия, %/мес на 1 s.d.", "мощность FM ln A_C", "мощность H-L A_C", "повторов"], prow)
    out["simulations"] = (t + f"\n\nКритерии `sim_checks.csv`: пройдено {V['sim.checks.pass']} из "
                          f"{V['sim.checks.n']}.\n" + fl + "\n\nМощность в мире V (`sim_power.csv`):\n\n" + pt)

    # upgrades table
    U = "results/upgrades"
    urows = [
        ["1. Amihud & Noh (2021): пропущенный член", "Прайсится ли DIF = ln A − ln|Ret| − ln A_C сверх A_C? Кто прав?",
         f"COV<0 в {V['an.covneg']} монето-месяцев; DIF col.3: {V['an.V1m']} (мес.), {V['an.V1y']} (12 мес.) против "
         f"A&N {V['an.p.col3']}; fDIF (12 мес.) {V['an.V4y']} против A&N {V['an.p.fdif']}; Res.lnA {V['an.V3m']} → "
         f"{V['an.V5m']} при ln|Ret|+IVOL (почти механически: R² Res.lnA на ln|Ret|+DIF = {V['an.r2_res_dif']})",
         f"данные не различают LS и A&N; единственный результат в сторону A&N — fDIF на 12-мес. мерах (ниже HLZ); "
         f"вердикт правила неустойчив к мин. числу монет ({V['an.sens.ret_adj']}), |t|≥3: {V['an.sens.n3']}",
         f"`{U}/amihud_noh/`"],
        ["2. λ по 5-мин. барам (Table 9/10)",
         "Объясняет ли price impact λ (знак из taker-объёма 5-мин. klines, сверено с aggTrades) премию?",
         f"{V['hf.months']} мес. ({V['hf.first']}…{V['hf.last']}); сверка с aggTrades на {V['hf.val.n']} монето-месяцах: "
         f"corr ln λ {V['hf.val.corr']}; corr(λ, A) {V['hf.corr.A']} (статья {V['hf.corr.A.paper']}); FM ln λ (1) "
         f"{V['hf.b1']} (статья {V['hf.b1.paper']}), (2) {V['hf.b2lam']} (статья {V['hf.b2lam.paper']}); с lnVOL "
         f"{V['hf.vol.b1']}; ln A_C в этом окне без λ {V['hf.b0']}; cost {V['hf.t10.c3']}, non-cost {V['hf.t10.n3']}",
         "λ хорошо мерит A и прайсится со знаком минус (проходит HLZ). Расходится со статьёй только в модели (1) "
         "(у статьи λ отдельно незначим); в модели (2) λ отрицателен и значим и у статьи. В этом окне сама премия "
         "A_C отсутствует, поэтому Table 9 не воспроизведена",
         f"`{U}/hf_lambda/`"],
        ["3. EDGE-спред и издержки", "Реализуема ли премия A_C после издержек? Объясняет ли спред A?",
         f"H-L A_C брутто {V['ed.gross']}, нетто {V['ed.net']}, с комиссией {V['ed.fee']} {V['ed.netfee']}; "
         f"безубыточная издержка {V['ed.breakeven']} на сторону; max |t| по EDGE-моделям {V['ed.maxabs_t']}",
         "дневной EDGE не мерит спред (вес на гэпе open−prev close); премия незначима и до издержек",
         f"`{U}/edge_spread/`"],
        ["4. Сентимент (Table 13)", "Больше ли премия после высокого Fear & Greed (mispricing)?",
         f"L−S {V['se.ls']} (статья {V['se.ls.paper']}); short {V['se.short']}; с CMKT(t−1) post hoc "
         f"{V['se.post.ls']}", "не подтверждено: знаки как в статье, значимости нет", f"`{U}/sentiment/`"],
        ["5. Рыночная неликвидность (Table 12)", "Растёт ли премия при высокой неликвидности рынка?",
         f"L−S {V['il.ls']} (статья {V['il.ls.paper']}); с CMKT(t−1) post hoc {V['il.post.ls']}; "
         f"медианный сплит {V['il.split']}",
         f"знак как в статье (против объяснения ликвидностью); сила зависит от спецификации: "
         f"{V['il.var_3']} из {V['il.var_n']} вариантов проходят 3", f"`{U}/sentiment/`"],
        ["6. MOEX TQBR (второй рынок, AT/AT_C)", "Повторяется ли декомпозиция на рынке акций РФ с turnover?",
         f"FM ln A_C (2) {V['mx.fm2']} (статья {V['mx.fm2.paper']}); Res.lnA (4) {V['mx.fm4res']}; "
         f"H-L A_C {V['mx.hlAC']}, альфа {V['mx.hlACalpha']}, нагрузка SMB {V['mx.smb']}",
         f"FM как в статье (объём прайсится, остаток нет); сортировки поглощаются размером, "
         f"{V['mx.topyear_share']} суммарного H-L — {V['mx.topyear']} год", f"`{U}/moex/`"],
        ["7. США после 2012 (OpenAP)", "Жива ли премия Illiquidity/DolVol после выборки статьи? Table A.1",
         f"Illiquidity {V['op.ill.pre']} → {V['op.ill.post']}; DolVol {V['op.dv.pre']} → {V['op.dv.post']}; "
         f"альфа Illiquidity на DolVol {V['op.pre.one']} (статья {V['op.paper.one']})",
         f"обе премии исчезли после 2012; разница значима на 1.96, по HLZ на грани "
         f"(t по лагам NW {V['op.ill.decay_lags']}: {V['op.ill.decay_ts']})", f"`{U}/openap_us/`"],
        ["8. Робастность и переобучение", "Насколько выводы держатся: бутстрап, годы, мощность, множественность",
         f"бутстрап-ДИ без нуля устойчиво: {V['rb.robust']} из {V['rb.n']} (оба Res.lnA); без 2021 H-L A_C "
         f"{V['x21.hlAC']}; MDE80 FM ln A_C {V['mde.fm2.slope']} на единицу ln A_C против наклона статьи "
         f"{V['mde.fm2.paper_slope']}",
         f"нуль неинформативен (MDE в {V['mde.fm2.slope_ratio']} раза больше наклона статьи); в базовом семействе "
         f"после Бонферрони/BH значимых нет", f"`{U}/robustness/`"],
    ]
    out["upgrades"] = md_table(["апгрейд", "вопрос", "результат (t — Newey–West, 6 лагов)", "вердикт", "папка"], urows)

    # overfitting
    lg_src = f"базовые {V['lg.base.3']}, " + V["lg.all.3_by_source"]
    out["overfitting"] = "\n".join([
        f"- **Предрегистрация.** Спецификация (`config.yaml`, `DEVIATIONS.md` D1–D11) зафиксирована в `{V['git.prereg']}`; "
        f"уточнения `DEVIATIONS.md` (D7 переписан, добавлены D12, D13) — в `{V['git.refine']}`, тоже до первого прогона "
        "на реальных данных "
        f"(`{V['git.firstrun']}`, git `{V['git.firstrun.sha']}`). `config.yaml` первого прогона {V['git.prereg.cfgmatch']} "
        f"с предрегистрированным (sha256 `{V['git.firstrun.cfg8']}…`). Изменения после первого прогона — только в журнале "
        "(J1 аудит данных, J2 винзоризация), с причиной.",
        f"- **Реестр тестов** (`results/upgrades/robustness/ledger_summary.csv`). Базовое семейство: "
        f"{V['lg.base.n']} уникальных тестов, |t|>1.96 у {V['lg.base.196']} (при нулевых эффектах ожидалось бы "
        f"{V['lg.base.exp']} на номинальном 5% и около {V['w0.exp_base']} при размере NW(6) из мира 0), "
        f"|t|>3 у {V['lg.base.3']}; после Бонферрони (порог |t| {V['lg.base.crit']}) — {V['lg.base.bonf']}, "
        f"после BH — {V['lg.base.bh']}. Базовые тесты и апгрейды вместе: {V['lg.all.n']} тестов, |t|>3 у "
        f"{V['lg.all.3']} (по источникам: {lg_src}), Бонферрони (порог |t| {V['lg.all.crit']}) {V['lg.all.bonf']}, "
        f"BH {V['lg.all.bh']}.",
        f"- **Размер теста.** В мире 0 NW(6) на наших N×T даёт ложные |t|>1.96 в {V['w0.fpr_min']}–{V['w0.fpr_max']} "
        f"повторов (|t|>3 — не более {V['w0.fpr3_max']}); эмпирический 5%-порог |t| {V['w0.crit_min']}–"
        f"{V['w0.crit_max']}. Критерии симуляций пройдены в {V['sim.checks.pass']} из {V['sim.checks.n']}; не пройдены: "
        f"{V['sim.checks.failed']}. Поэтому пункт P0 «в мире 0 около 5%» буквально не выполнен, и рядом с 1.96 мы всегда "
        "показываем эмпирический порог мира 0 и порог HLZ 3.",
        f"- **Бутстрап по блокам месяцев** ({V['rb.block']} мес., {V['rb.draws']} повторов, ещё {V['rb.seeds']} "
        f"сидов): ДИ без нуля у {V['rb.excl']} из {V['rb.n']} статистик, устойчиво к сиду — у {V['rb.robust']} "
        f"(FM Res.lnA (3) {V['rb.ci.fm3_ResLnA']}, (4) {V['rb.ci.fm4_ResLnA']}); FM ln A_C (4) на грани "
        f"({V['rb.seedshare.fm4_lnA_C']} сидов без нуля).",
        f"- **Плацебо** (`placebo.csv`): перемешивание мер внутри месяца ({V['plc.shuffle_reps']} перестановок): средний "
        f"t по перестановкам не больше {V['plc.shuffle_maxabs_meant']} по модулю, s.d. t между перестановками "
        f"{V['plc.shuffle_sd_t_min']}–{V['plc.shuffle_sd_t_max']}, доля |t|>1.96 не выше {V['plc.shuffle_share196_max']}; "
        f"подглядывание (лаг 0 вместо t−2) даёт |t| {V['plc.leak_minabs_t']}–{V['plc.leak_maxabs_t']}, то есть ошибка "
        "лага была бы видна.",
        f"- **Устойчивость.** Без 2021 года H-L A_C {V['x21.hlAC']}, FM ln A_C {V['x21.fm2']}, Res.lnA (3) "
        f"{V['x21.fm3']}; без одного года t H-L A_C в диапазоне {V['loyo.hlAC']}. Винзоризация доходностей, сырая "
        f"доходность, взвешивание и подпериоды: {V['rob.same']} из {V['rob.n']} вариантов сохраняют базовый вывод "
        "(`robustness_vs_base.csv`).",
        f"- **Мощность.** MDE80 для H-L A_C {V['mde.hlAC']}%/мес против премии статьи {V['mde.hlAC.paper']}. Для FM "
        f"ln A_C: {V['mde.fm2.slope']} на единицу ln A_C против наклона статьи {V['mde.fm2.paper_slope']} (в "
        f"{V['mde.fm2.slope_ratio']} раза больше); на 1 s.d. — {V['mde.fm2']} (наше s.d. {V['mde.fm2.sd']}) против "
        f"{V['mde.fm2.paper']} у статьи (её s.d. {V['mde.fm2.paper_sd']}, с. 13), или {V['mde.fm2.paper_oursd']}, если "
        "наклон статьи умножить на наше s.d. Незначимость здесь не доказывает отсутствие премии размера статьи.",
        "- **Survivorship и look-ahead.** Вселенная — все USDT-пары из листинга `data.binance.vision`, включая "
        "делистинг. Сигнал строго из t−2: `tests/test_alignment.py` независимо пересчитывает все лаговые величины "
        "панели по сырым ценам, а плацебо с лагом 0 (выше) показывает, что утечка дала бы заметные |t|.",
        "- **Post hoc варианты** (добавлены после того, как увидели основной результат; не предрегистрированы) помечены "
        "в CSV и SUMMARY каждого апгрейда и никогда не являются основной спецификацией.",
    ])

    # runtime and disk
    rrows = [["`make run` (базовый прогон)", fsec(B["manifest"]["timings_sec"]["total"]), f"`{V['base.dir']}/run_manifest.json`"],
             ["`make sims` (симуляции, параллельно)", V["sim.wall"], "`results/simulations/sim_manifest.json`"]]
    rep = rd / "report_manifest.json"
    if rep.exists():
        rm = V.json(rep)
        rrows.append(["`make report`", fsec(sum(rm["timings_sec"].values())), f"`{V['base.dir']}/report_manifest.json`"])
    for k, m in sorted(up.items()):
        ts = m.get("timings_sec", {})
        tot = ts.get("total", sum(v for v in ts.values() if isinstance(v, (int, float))))
        rrows.append([f"`make upgrade-{k}`", fsec(tot), f"`results/upgrades/{k}/run_manifest.json`"])
    rt = md_table(["этап", "секунд (последний прогон, кэш данных есть)", "источник"], rrows)
    raw = ROOT / "data" / "raw"
    drows = []
    if raw.exists():
        for p in sorted(raw.iterdir()):
            if p.is_dir():
                size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                drows.append([f"`data/raw/{p.name}/`", fx(size / 1e6, 1)])
    dt = (md_table(["папка", "МБ (10^6 байт) на машине автора"], drows) if drows
          else "_data/raw/ отсутствует на этой машине._")
    out["runtime"] = (rt + "\n\nВремя загрузки данных в манифесты не пишется: оно зависит от сети. Объём сырых "
                      "данных (считается при сборке документации):\n\n" + dt)

    # manifest fields
    man = B["manifest"]
    out["manifest_fields"] = ("Поля `run_manifest.json` базового прогона: " + ", ".join(f"`{k}`" for k in man)
                              + ".")

    # slide 8 compact table
    pvu = B["pvu"].set_index("id")
    sel = ["corrAAC", "QratioAC", "corrAabs", "HLArawCoef", "HLACrawCoef", "HLACalphaCoef", "HLResArawCoef",
           "FMlnA", "FMlnAC", "FMResLnA", "FMResLnAfour", "IMLAalpha"]
    srows = [[pvu.loc[i, "metric"], V[f"pvu.{i}.paper"], V[f"pvu.{i}"], pvu.loc[i, "match"]] for i in sel]
    T = B.setdefault("tables", {})
    T["slide8"] = (["метрика", "статья (NYSE/AMEX 1964–2012)", "мы (Binance)", "совпадение"], srows)

    # slide 9: lambda (paper: Table 9 is NYSE/AMEX 1983-2012, Table 10 is 1964-2012)
    T["slide9"] = (["", "статья (T9: 1983–2012; T10: 1964–2012)",
                    f"мы (Binance {V['hf.first']}…{V['hf.last']}, 5-мин. бары)"], [
        ["corr(λ, A)", V["hf.corr.A.paper"], V["hf.corr.A"]],
        ["corr(λ, A_C)", V["hf.corr.A_C.paper"], V["hf.corr.A_C"]],
        ["corr(λ, Res.A)", V["hf.corr.ResA.paper"], V["hf.corr.ResA"]],
        ["FM ln λ, модель (1)", V["hf.b1.paper"], V["hf.b1"]],
        ["FM ln λ при ln A_C, модель (2)", V["hf.b2lam.paper"], V["hf.b2lam"]],
        ["FM ln A_C при ln λ, модель (2)", V["hf.b2ac.paper"], V["hf.b2ac"]],
        ["T10: cost-часть, модель (3)", V["hf.t10.c3.paper"], V["hf.t10.c3"]],
        ["T10: non-cost-часть, модель (3)", V["hf.t10.n3.paper"], V["hf.t10.n3"]],
    ])

    # slide 10: mispricing. Paper columns of Tables 11-13 are turnover sorts (AT_C) with Baker-Wurgler /
    # Pastor-Stambaugh states; ours are A_C with Fear & Greed / AR(2) of EW A: only sign and t compare
    T["slide10"] = (["тест", "статья (AT_C; BW / PS)", "мы (A_C; Fear & Greed / AR(2) по A)"], [
        ["Сентимент → L−S (Table 13)", V["se.ls.paper"], V["se.ls"]],
        ["Сентимент → short-нога", V["se.short.paper"], V["se.short"]],
        ["Неликвидность рынка → L−S (Table 12)", V["il.ls.paper"], V["il.ls"]],
        ["то же, + CMKT(t−1), post hoc", "—", V["il.post.ls"]],
        ["Январь: наклон (Table 11; статья — ln AT_C)", "в январе премии объёма нет",
         f"январь {V['t7.jan']}, n {V['t7.jan.n']}"],
        ["Остальные месяцы", "вне января премия объёма сильная", f"{V['t7.rest']}, n {V['t7.rest.n']}"],
    ])
    for k, (h, r) in T.items():
        out[k] = md_table(h, r)
    out["slide10"] += ("\n\n_Сравнимы знак и t, не масштаб: у статьи сортировки по turnover (AT_C) и индексы "
                       "Baker–Wurgler / Pástor–Stambaugh, у нас — A_C, Fear & Greed и AR(2)-остаток EW-среднего A._")

    # team and repository (README header and title slide): from slides/meta.yaml, filled by the team
    meta = B.get("meta", {})
    team = ", ".join(meta.get("team") or [])
    repo = meta.get("repo_url") or ""
    out["team"] = (f"**Состав команды:** {team or '_не заполнено: `slides/meta.yaml` → team_'}  \n"
                   f"**Репозиторий:** {repo or '_не заполнено: `slides/meta.yaml` → repo_url_'}")
    return out


# ------------------------------------------------------------------------------------ LaTeX for the deck
TEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "`": ""}


def tex(x) -> str:
    """Escape a value for LaTeX; a hyphen before a digit that is not part of a date becomes a real minus."""
    out = "".join(TEX_ESC.get(ch, ch) for ch in str(x))
    return re.sub(r"(?<![\w.])-(?=\d)", "\u2212", out)


def tex_table(header: list[str], rows: list[list[str]], widths: list[float]) -> str:
    spec = "".join(f"p{{{w}\\linewidth}}" for w in widths)
    lines = [f"\\begin{{tabular}}{{@{{}}{spec}@{{}}}}", "\\toprule",
             " & ".join(f"\\textbf{{{tex(h)}}}" for h in header) + " \\\\", "\\midrule"]
    lines += [" & ".join(tex(c) for c in r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


# column widths as shares of \\linewidth; with the column separation they add up to about 1
TEX_WIDTHS = {"slide8": [0.38, 0.21, 0.21, 0.11], "slide9": [0.33, 0.30, 0.31],
              "slide10": [0.33, 0.29, 0.32], "sims": [0.05, 0.16, 0.33, 0.12, 0.09, 0.09]}


def tex_outputs(V: Values, B: dict) -> dict[Path, str]:
    """slides/gen/values.tex (\\V{key} for every key) and one file per slide table."""
    meta = B.get("meta", {})
    head = ["% generated by src/build_docs.py from results/: do not edit", ""]
    vals = [f"\\expandafter\\def\\csname v@{k}\\endcsname{{{tex(V[k])}}}" for k in sorted(V)]
    extra = [f"\\def\\basedir{{../{V['base.dir']}}}",
             f"\\def\\teamnames{{{tex(', '.join(meta.get('team') or [])) or 'Team 12'}}}",
             f"\\def\\repourl{{{tex(meta.get('repo_url') or '')}}}"]
    out = {GEN / "values.tex": "\n".join(head + vals + extra) + "\n"}
    for name, (h, rows) in B.get("tables", {}).items():
        out[GEN / f"{name}.tex"] = "% generated by src/build_docs.py: do not edit\n" + tex_table(h, rows, TEX_WIDTHS[name])
    return out


def deck_hand_typed(text: str) -> list[str]:
    """Decimal numbers in slides/deck.tex outside \\V{...}, comments and file names."""
    t = re.sub(r"(?<!\\)%.*", "", text)
    t = re.sub(r"\\V\{[^}]*\}", "", t)
    t = re.sub(r"\\(includegraphics|input|url|href)(\[[^]]*\])?\{[^}]*\}", "", t)
    # layout lengths (column widths, array stretch) are not results
    t = re.sub(r"\d*\.\d+\\(textwidth|linewidth|textheight)", "", t)
    t = re.sub(r"\\arraystretch\}?\{[^}]*\}", "", t)
    out = []
    for ln in t.splitlines():
        for m in DEC_RE.finditer(ln):
            if m.group(0) not in CONSTANTS:
                out.append(f"{m.group(0)!r} in: {ln.strip()[:120]}")
    return out


def deck_keys(text: str) -> set[str]:
    t = re.sub(r"(?<!\\)%.*", "", text)
    return set(re.findall(r"\\V\{([^}]*)\}", t))


# ------------------------------------------------------------------------------------ render
def render(text: str, V: Values, blk: dict[str, str], missing: set[str]) -> str:
    def inl(m):
        k = m.group(1)
        if k not in V:
            missing.add(k)
            return m.group(0)
        return f"<!--v:{k}-->{V[k]}<!--/v-->"
    text = PLACEHOLDER_RE.sub(inl, text)
    text = INLINE_RE.sub(inl, text)

    def bl(m):
        k = m.group(2)
        if k not in blk:
            missing.add("BLOCK:" + k)
            return m.group(0)
        return f"{m.group(1)}\n{blk[k]}\n{m.group(4)}"
    return BLOCK_RE.sub(bl, text)


DEC_RE = re.compile(r"(?<![\w.\-/])-?\d+\.\d+(?![\w.])")
CONSTANTS = {"1.96"}  # the 5% two-sided normal critical value is a definition, not a result


def hand_typed(text: str) -> list[str]:
    """Decimal numbers outside markers, code and comments: candidates for hand-typed numbers."""
    t = BLOCK_RE.sub("", text)
    t = INLINE_RE.sub("", t)
    t = re.sub(r"```.*?```", "", t, flags=re.S)
    t = re.sub(r"`[^`\n]*`", "", t)
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"\]\([^)]*\)", "]", t)
    out = []
    for ln in t.splitlines():
        for m in DEC_RE.finditer(ln):
            if m.group(0) in CONSTANTS:
                continue
            out.append(f"{m.group(0)!r} in: {ln.strip()[:120]}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    R = Path(a.results_dir).resolve()
    U = R / "upgrades"
    V, B = Values(), {}
    base_values(V, B, R)
    sim_values(V, B, R)
    an_values(V, B, U)
    hf_values(V, B, U)
    edge_values(V, B, U)
    sentiment_values(V, B, U)
    openap_values(V, B, U)
    moex_values(V, B, U)
    robustness_values(V, B, U)
    pre = subprocess.run(["git", "log", "--diff-filter=A", "--format=%h", "--", "config.yaml"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    B["prereg"] = pre[-1] if pre else "n/a"
    V.put("git.prereg", B["prereg"])
    prereg_values(V, B, R)
    mp = ROOT / "slides" / "meta.yaml"
    B["meta"] = (yaml.safe_load(V.text(mp)) or {}) if mp.exists() else {}
    blk = blocks(V, B, R)
    if a.list:
        for k in sorted(V):
            print(f"{k} = {V[k]}")
        print("blocks:", ", ".join(sorted(blk)))
        return 0

    rc, outputs = 0, {}
    for doc in DOCS:
        if not doc.exists():
            print(f"skip (missing): {doc.relative_to(ROOT)}")
            continue
        old = doc.read_text()
        missing: set[str] = set()
        new = render(old, V, blk, missing)
        if missing:
            print(f"{doc.relative_to(ROOT)}: unknown keys {sorted(missing)}", file=sys.stderr)
            rc = 1
        ht = hand_typed(new)
        if ht:
            print(f"{doc.relative_to(ROOT)}: {len(ht)} decimal number(s) outside markers:", file=sys.stderr)
            for h in ht:
                print("   ", h, file=sys.stderr)
            if a.strict:
                rc = 1
        if a.check:
            if new != old:
                print(f"out of date: {doc.relative_to(ROOT)} (run `make docs`)", file=sys.stderr)
                rc = 1
        elif new != old:
            doc.write_text(new)
            print(f"wrote {doc.relative_to(ROOT)}")
        else:
            print(f"up to date: {doc.relative_to(ROOT)}")
        outputs[str(doc.relative_to(ROOT))] = hashlib.sha256(new.encode()).hexdigest()
    # the slide deck: generated values and tables, plus checks of the hand-written deck.tex
    for path, content in tex_outputs(V, B).items():
        rel = str(path.relative_to(ROOT))
        old = path.read_text() if path.exists() else None
        if a.check:
            if old != content:
                print(f"out of date: {rel} (run `make docs`)", file=sys.stderr)
                rc = 1
        elif old != content:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            print(f"wrote {rel}")
        outputs[rel] = hashlib.sha256(content.encode()).hexdigest()
    if DECK.exists():
        deck = DECK.read_text()
        unknown = sorted(deck_keys(deck) - set(V))
        if unknown:
            print(f"slides/deck.tex: unknown keys {unknown}", file=sys.stderr)
            rc = 1
        ht = deck_hand_typed(deck)
        if ht:
            print(f"slides/deck.tex: {len(ht)} decimal number(s) outside \\V{{}}:", file=sys.stderr)
            for h in ht:
                print("   ", h, file=sys.stderr)
            if a.strict:
                rc = 1
    if not a.check and rc == 0:
        man = {"stage": "docs", "generated_by": "src/build_docs.py", "base_run_id": V["base.run"],
               "upgrade_run_ids": {k: m["run_id"] for k, m in sorted(B["upgrade_manifests"].items())},
               "created_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), "git": git_state(),
               "code_sha256": sha256(Path(__file__)), "n_values": len(V), "inputs_sha256": V.inputs,
               "outputs_sha256": outputs, "timings_sec": {"total": round(time.time() - t0, 3)}}
        (R / "docs_manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
