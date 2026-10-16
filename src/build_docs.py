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
        --strict  exit 1 if a decimal number appears outside markers (a hand-typed number), if a
                  deliverable contains Cyrillic text, or if the team list in slides/meta.yaml is empty.
                  The hand-typed check covers decimals only: integers outside markers are not checked.
        --list    print all keys with their values and exit
Writes: README.md, slides/PLAN.md, slides/gen/*.tex, results/docs_manifest.json (inputs with sha256, git state).
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


def deck_roles(tex: str) -> dict:
    """Speaker per main slide and Q&A owner per backup slide, from the \\speaker{} / \\owner{} tags of deck.tex.
    Main slides are numbered by the '% ---- N' separators; backup slides by their 'B<k>.' frame titles."""
    out = {"speaker": {}, "owner": {}}
    label, who, kind = None, None, None
    for line in tex.splitlines():
        m = re.match(r"%\s*-{5,}\s*(\d+)\s*$", line)
        if m:
            label = m.group(1)
        m = re.match(r"\\(speaker|owner)\{([^}]*)\}", line.strip())
        if m:
            kind, who = m.group(1), m.group(2)
            if kind == "speaker" and label:
                out["speaker"].setdefault(who, []).append(label)
                label = None
        m = re.match(r"\\begin\{frame\}\{(B\d+)\.", line.strip())
        if m and kind == "owner" and who:
            out["owner"].setdefault(who, []).append(m.group(1))
    return out


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
    # same scope as the run manifests (src/run.py git_info): code and config only, generated outputs excluded
    return {"sha": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", *CODE_PATHS)),
            "dirty_scope": "code+config (" + ", ".join(CODE_PATHS) + "); generated outputs excluded"}


def commit_with_code_hash(code_sha: str, prefer: str | None = None) -> str | None:
    """Short SHA of a commit whose src/*.py hash (as run.code_hash computes it) equals code_sha, else None.
    `prefer` (the commit the manifest records) is tried first; later commits with the same src/ come after it."""
    def run(*args, text=True):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=text).stdout
    for c in ([prefer] if prefer else []) + run("rev-list", "--all").split():
        names = sorted(n for n in run("ls-tree", "--name-only", c, "src/").split()
                       if n.endswith(".py") and n.count("/") == 1)
        h = hashlib.sha256()
        for n in names:
            h.update(Path(n).name.encode())
            h.update(run("show", f"{c}:{n}", text=False))
        if h.hexdigest() == code_sha:
            return c[:7]
    return None


# files written by `make docs slides` themselves: they are regenerated on every build, so counting them as
# "uncommitted" would make the count depend on the build itself (they must be committed together with the rest)
DOC_OUTPUTS = ("README.md", "slides/PLAN.md", "slides/gen/", "results/docs_manifest.json", "slides/paper15_team12_")


def uncommitted() -> tuple[list[str], list[str]]:
    """(modified tracked files, untracked files) of the whole working tree against HEAD, without DOC_OUTPUTS."""
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
                         capture_output=True, text=True).stdout
    mod, new = [], []
    for ln in out.splitlines():
        if not ln.strip():
            continue
        path = ln[3:].strip().strip('"')
        if path.startswith(DOC_OUTPUTS):
            continue
        (new if ln.startswith("??") else mod).append(path)
    return sorted(mod), sorted(new)


def uncommitted_text(mod: list[str], new: list[str]) -> str:
    """One line: counts by top-level folder and the untracked files (a folder with several new files is shown once)."""
    if not mod and not new:
        return "none (the working tree matches the last commit, apart from the generated documents)"

    def top(p):
        return p.split("/", 1)[0] + ("/" if "/" in p else "")
    by = {}
    for p in mod + new:
        by[top(p)] = by.get(top(p), 0) + 1
    shown = {}
    for p in new:
        d = p.rsplit("/", 1)[0] + "/" if "/" in p else p
        shown.setdefault(d, []).append(p)
    nl = [f"`{d}` ({len(v)} files)" if len(v) > 1 else f"`{v[0]}`" for d, v in sorted(shown.items())]
    return (f"{len(mod) + len(new)} files ({len(mod)} modified, {len(new)} untracked; by folder: "
            + ", ".join(f"{k} {v}" for k, v in sorted(by.items())) + ")"
            + ("; untracked: " + ", ".join(nl) if nl else ""))


# code state that a pytest log refers to: tracked changes against HEAD plus untracked files, code paths only
CODE_PATHS = ["src", "tests", "config.yaml", "Makefile", "mk", "requirements.txt"]


def code_state_hash() -> str:
    h = hashlib.sha256()
    h.update(subprocess.run(["git", "diff", "HEAD", "--", *CODE_PATHS], cwd=ROOT, capture_output=True).stdout)
    new = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "--", *CODE_PATHS], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    for f in sorted(new):
        h.update(f.encode())
        h.update((ROOT / f).read_bytes())
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return head[:7] + "+" + h.hexdigest()[:16]


def deviation_ranges(path: Path) -> str:
    """'D1–D18, U1–U8, J1–J2' from the ids in DEVIATIONS.md."""
    txt = path.read_text() if path.exists() else ""
    parts = []
    for prefix, pat in [("D", r"^\| D(\d+) \|"), ("U", r"^\| U(\d+) \|"), ("J", r"^### J(\d+) ")]:
        ids = sorted({int(x) for x in re.findall(pat, txt, re.M)})
        parts.append(f"{prefix}{ids[0]}–{prefix}{ids[-1]}" if ids else f"{prefix}?")
    return ", ".join(parts)


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
    V.put("base.code8", man["code_sha256"][:8])
    hit = commit_with_code_hash(man["code_sha256"], prefer=man["git"]["sha"])
    V.put("base.code_commit", f"matches commit {hit}" if hit else "matches no commit in the git history")
    V.put("base.code_checkout", f"so the code that wrote it can be checked out (`git checkout {hit}`)" if hit
          else "so the exact code that wrote it cannot be checked out from git")

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
        if pd.notna(getattr(r, "us_mde80", float("nan"))):
            V.put(f"pvu.{r.id}.mde", fx(r.us_mde80, 2))
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
    # 'yes' also covers 'both insignificant', which can have opposite signs (the IMLA_C alphas)
    yes_opp = ls[(ls.match == "yes") & (ls.us * ls.paper < 0)]
    V.put("sc.yes_oppsign", len(yes_opp))
    V.put("sc.yes_oppsign_rows", "; ".join(yes_opp.metric) if len(yes_opp) else "none")
    res = ls[ls.id.str.contains("ResLnA")]
    B["res_rows"] = res
    V.put("sc.res_n", len(res))
    # |t| >= 1.96 regardless of the paper (includes rows where the paper itself is not significant)
    V.put("sc.res_sig", int((res.us_t.abs() >= 1.96).sum()))
    res_ps = res[res.paper_t.abs() >= 1.96]
    V.put("sc.res_paper_sig", len(res_ps))
    # the paper's significance repeated: paper significant and match == 'yes' (same sign, |t| >= 1.96)
    V.put("sc.res_rep196", int((res_ps.match == "yes").sum()))
    rep = ls[(ls.paper_t.abs() >= 1.96) & (ls.match == "yes")]
    V.put("sc.rep196", len(rep))
    V.put("sc.paper_sig", int((ls.paper_t.abs() >= 1.96).sum()))
    oth = rep[~rep.id.str.contains("ResLnA")]
    V.put("sc.rep196_other_n", len(oth))
    V.put("sc.rep196_other_rows", "; ".join(f"{r.metric} (t {fx(r.us_t)})" for r in oth.itertuples())
          if len(oth) else "none")
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
    # 'same conclusion' = same sign and same side of 1.96: split it into 'insignificant stays insignificant'
    # and 'significant stays significant', so the count is not read as robustness of significant results
    rob_yes = rob[rob.same_conclusion == "yes"]
    V.put("rob.same_insig", int((rob_yes.base_t.abs() < 1.96).sum()))
    V.put("rob.same_sig", int((rob_yes.base_t.abs() >= 1.96).sum()))
    rsig = rob[rob.base_t.abs() >= 1.96]
    V.put("rob.sig_n", len(rsig))
    V.put("rob.sig_stats", ", ".join(dict.fromkeys(rsig.statistic)) or "none")
    V.put("rob.variants", "; ".join(dict.fromkeys(rob.variant)))
    V.put("rob.posthoc_n", int(rob.variant.str.contains("post hoc").sum()))
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
            why = f"rule ≥ {pc(float(ge.group(1)))}"
        elif ci:
            why = f"95% CI {pc(float(ci.group(1)), 1)}–{pc(float(ci.group(2)), 1)} excludes 5%"
        else:
            why = f"rule {r.rule}"
        fails.append(f"world {r.world}, {stat}: {pc(r.value, 1)} ({why})")
    V.put("sim.checks.failed", "; ".join(fails) if fails else "none")
    # criteria of the form 'share <= x' for signals that should be unpriced: say how loose they are
    rule = chk.rule.astype(str).str.strip()
    loose = chk[rule.str.startswith("<=")]
    V.put("sim.checks.loose_n", len(loose))
    V.put("sim.checks.loose_rule", "≤ " + ", ".join(dict.fromkeys(
        pc(float(re.search(r"[0-9.]+", x).group(0))) for x in loose.rule.astype(str))) if len(loose) else "none")
    V.put("sim.checks.loose_max", pc(loose.value.max()) if len(loose) else "n/a")
    strict = passed & ~(rule.str.startswith("<=") & (chk.value.astype(float) > 0.05))
    V.put("sim.checks.pass_strict5", int(strict.sum()))
    pw = V.csv(sd / "sim_power.csv")
    B["sim_power"] = pw
    for prem in sorted(pw.premium.unique()):
        for stat, key in [("fm2_lnA_C", "fm2"), ("hl_A_C", "hlAC")]:
            r = one(pw, premium=prem, stat=stat)
            V.put(f"pow.{fx(100 * prem, 2)}.{key}", pc(r["share_t_gt_1.96"]))
            # binomial 95% CI written by simulate.py for the share of |t| > 1.96 (n_reps per point)
            V.put(f"pow.{fx(100 * prem, 2)}.{key}.ci", f"{pc(r.ci95_lo)}–{pc(r.ci95_hi)}")
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
    for test, sample, sk in [("V2", "monthly", "m"), ("V4", "12m", "y")]:
        r = vd[(vd.test == test) & (vd["sample"] == sample) & (vd.dep == "ret")]
        if len(r) == 1:
            V.put(f"an.ret.{test}{sk}", ct(r.iloc[0].coef, r.iloc[0].t))
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
    fd = so[(so.test == "V4") & (so.dep == "ret_adj")]
    V.put("an.sens.fdif_t", " / ".join(fx(one(fd, min_obs=o).t, 2) for o in order))
    V.put("an.sens.n3", int((so.t.abs() >= 3).sum()))
    V.put("an.minN", int(so.min_n.min()))
    iv = V.csv(d / "ivol_channel.csv").set_index("sample")
    V.put("an.corr_res_absret", fx(iv.loc["monthly", "corr_res_ln_absret"], 2))
    V.put("an.r2_res", fx(iv.loc["monthly", "r2_res_on_ln_absret_ivol"], 2))
    V.put("an.corr_res_dif", fx(iv.loc["monthly", "corr_res_perp_DIF_perp"], 2))
    V.put("an.r2_res_abs", fx(iv.loc["monthly", "r2_res_on_ln_absret"], 2))
    V.put("an.r2_res_dif", fx(iv.loc["monthly", "r2_res_on_ln_absret_DIF"], 2))
    from amihud_noh import scale_ratios  # same pairs as the key-regressions table of the A&N SUMMARY.md
    rat = scale_ratios(V.csv(d / "fm.csv"), pr)
    V.put("an.scale_min", fx(rat.ratio.min(), 1))
    V.put("an.scale_max", fx(rat.ratio.max(), 0))
    V.put("an.scale_lt10", f"{int((rat.ratio < 10).sum())} of {len(rat)}")
    V.put("an.scale_opp", int(rat.opposite_sign.sum()))
    eg = rat[(rat["sample"] == "monthly") & rat["var"].isin(["lnA_C", "ln_absret"])]
    V.put("an.scale_eg", "; ".join(f"{'ln A_C' if r.var == 'lnA_C' else 'ln|Ret|'} {fa(r.ours)} vs {fp(r.paper)}"
                                    for r in eg.itertuples()))
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
        r = one(dg[dg.variant.str.startswith(pref)], model="(0)", var="lnA_C")   # A_C alone, no λ
        V.put(f"hf.{key}.b0", ct(r.coef, r.t))
    lam = dg[(dg["var"] == "lnlam")]
    V.put("hf.lam_tmin", fx(lam.t.abs().min(), 2))
    V.put("hf.lam_tmax", fx(lam.t.abs().max(), 2))
    vs = V.csv(d / "validation_summary.csv")
    r = one(vs, pair="exact_vs_kl_sqrt_sides", months="all")
    V.put("hf.val.corr", fx(r.pearson_log, 2))
    V.put("hf.val.n", int(r.n))
    V.put("hf.val.spearman", fx(r.spearman, 2))
    V.put("hf.val.ratio", fx(r.median_ratio, 1))
    vm = vs[(vs.pair == "exact_vs_kl_sqrt_sides") & (vs.months != "all")]
    V.put("hf.val.months", ", ".join(vm.months))
    V.put("hf.val.n_months", len(vm))
    ex = dg[dg.variant.str.startswith("ex exit month")]
    V.put("hf.exit.b1", ct(one(ex, model="(1)", var="lnlam").coef, one(ex, model="(1)", var="lnlam").t))
    V.put("hf.exit.b2lam", ct(one(ex, model="(2)", var="lnlam").coef, one(ex, model="(2)", var="lnlam").t))
    iv = dg[dg.variant.str.startswith("+ ivol")]
    V.put("hf.ivol.b1", ct(one(iv, model="(1)", var="lnlam").coef, one(iv, model="(1)", var="lnlam").t))
    r = one(vol, model="(0)", var="lnA_C")
    V.put("hf.vol.b0ac", ct(r.coef, r.t))
    V.put("hf.corr.ATC.paper", fx(B["paper"]["lou_shu_2017"]["table9"]["corr_ATC_lambda"], 3))
    # paper's January / other-month slopes (Table 8: ln λ; Table 11: ln AT_C and ln λ), typed from the paper
    # into src/paper_numbers.yaml; used in words on slides 9-10
    ls = B["paper"]["lou_shu_2017"]
    for key, tab, name in [("t8.jan_lam", "table8", "jan_ln_lambda"), ("t8.rest_lam", "table8", "nonjan_ln_lambda"),
                           ("t11.jan_atc", "table11", "jan_ln_ATC"), ("t11.rest_atc", "table11", "nonjan_ln_ATC"),
                           ("t11.jan_lam2", "table11", "jan_ln_lambda_m2"),
                           ("t11.rest_lam2", "table11", "nonjan_ln_lambda_m2"),
                           ("t11.jan_lam4", "table11", "jan_ln_lambda_m4"),
                           ("t11.rest_lam4", "table11", "nonjan_ln_lambda_m4")]:
        x = ls[tab][name]
        V.put(f"{key}.paper", f"{fp(x['coef'])} (t {fx(x['t'])})")


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
    r = one(pb, state="illiq_ar2_log", controls="CMKT+CMOM", portfolio="Long-Short")
    V.put("il.log", ct(r.b, r.t))
    r = one(pb, state="illiq_ar2", controls="CMKT", portfolio="Long-Short")
    V.put("il.cmkt_only", ct(r.b, r.t))
    V.put("il.main_controls", one(pb[(pb.state == "illiq_ar2") & pb.main.astype(bool)], portfolio="Long-Short").controls)


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
        r = one(sp, dep="Illiquidity", period=per, spec="ff4_only")
        V.put(f"op.{pk}.smb", ct(r.b_SMB, r.t_SMB))
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
    r = one(pv, item="Quintile H-L on ResA, raw (%/mo)")
    V.put("mx.hlResA", ct(r.moex, r.moex_t))
    V.put("mx.hlResA.paper", f"{fp(r.paper)} (t {fx(r.paper_t)})")
    hl = pv[pv["item"].str.startswith("Quintile H-L on A_C, alpha")].iloc[0]
    V.put("mx.hlACalpha", ct(hl.moex, hl.moex_t))
    hl3 = pv[pv.moex_sig_hlz3.astype(str).str.lower() == "true"]
    V.put("mx.hlz_list", "; ".join(f"{r.item} t {fx(r.moex_t)}" for r in hl3.itertuples()))
    V.put("mx.hlz_n", len(hl3))
    t4 = V.csv(d / "T4_fm.csv")
    t4m = t4[(t4.variant == "main") & t4["var"].isin(["lnA", "lnA_C", "lnAT", "lnAT_C"])]
    V.put("mx.vol_n", len(t4m))
    V.put("mx.vol_196", int((t4m.t > 1.96).sum()))
    r = one(t4, variant="main, raw returns", family="A (RUB volume)", model="(4)", var="lnA_C")
    V.put("mx.raw.fm4ac", ct(r.coef, r.t))
    r = one(t4, variant="main, raw returns", family="A (RUB volume)", model="(2)", var="lnA_C")
    V.put("mx.raw.fm2ac", ct(r.coef, r.t))
    r = one(t4, variant="diagnostic: raw returns, months of ret_adj FM", family="A (RUB volume)", model="(4)",
            var="lnA_C")
    V.put("mx.rawsame.fm4ac", ct(r.coef, r.t))
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
        V.put(f"lg.{k}.exp", fx(r["expected_false_1.96_if_all_null"], 2))
        V.put(f"lg.{k}.bonf", int(r.n_bonferroni_sig))
        V.put(f"lg.{k}.bh", int(r.n_bh_sig))
        V.put(f"lg.{k}.crit", fx(r.bonferroni_crit_t, 2))
        V.put(f"lg.{k}.rows", int(r.n_rows))
    tests = lg[(lg.role == "test") & lg.source.str.startswith("upgrade:")]
    V.put("lg.all.3_by_source", ", ".join(f"{r.source.split(':', 1)[1]} {int(r.n_abs_t_gt_3)}"
                                         for r in tests.itertuples() if r.n_abs_t_gt_3 > 0))
    led = V.csv(d / "ledger.csv")
    surv = led[led.in_family.astype(bool) & (led.p_bonferroni < 0.05)]

    def kind(r):
        if str(r.t_column).startswith("load_"):
            return "factor loading"
        m = re.search(r"var=([^|]+)", str(r.test)) or re.search(r"portfolio=([^|]+)", str(r.test))
        k = m.group(1).strip() if m else str(r.t_column)
        return {"lnlam": "ln λ", "cost": "T10 cost part"}.get(k, k)
    parts = []
    for src, g in surv.groupby("source", sort=True):
        kinds = g.apply(kind, axis=1).value_counts()
        parts.append(f"{src.split(':', 1)[-1]}: " + ", ".join(f"{k} {n}" for k, n in kinds.items()))
    V.put("lg.all.bonf_by_kind", "; ".join(parts) if parts else "none")
    V.put("lg.all.bonf_loadings", int(surv.t_column.astype(str).str.startswith("load_").sum()))

    def field(test, name):
        m = re.search(name + r"=([^|]+)", str(test))
        return m.group(1).strip() if m else "?"
    # Bonferroni survivors from sentiment/panel_b.csv, described by their own state and controls
    ss = surv[surv.source == "upgrade:sentiment"]
    V.put("lg.all.bonf_sent", "; ".join(f"{field(r.test, 'state')} with controls {field(r.test, 'controls')}, "
                                        f"{field(r.test, 'portfolio')}" for r in ss.itertuples()) or "none")
    V.put("lg.all.bonf_sent_n", len(ss))
    # hf_lambda survivors: which windows / variants they come from
    hs = surv[surv.source == "upgrade:hf_lambda"]
    V.put("lg.all.bonf_hf_n", len(hs))
    hk = hs.apply(kind, axis=1) if len(hs) else pd.Series(dtype=str)
    V.put("lg.all.bonf_hf_lam_n", int(hk.isin(["ln λ", "T10 cost part"]).sum()))
    oth = hk[~hk.isin(["ln λ", "T10 cost part"])].value_counts()
    V.put("lg.all.bonf_hf_other", ", ".join(f"{n} {k} row{'s' if n > 1 else ''}" for k, n in oth.items()) or "none")
    V.put("lg.all.bonf_hf", "; ".join(f"{kind(r)} ({field(r.test, 'variant') if 'variant=' in str(r.test) else field(r.test, 'spec') + ', ' + field(r.test, 'cost_benchmarks')}, model {field(r.test, 'model')})"
                                      for r in hs.itertuples()) or "none")
    tests = led[(led.role == "test") & led.unique.astype(bool)]
    V.put("lg.all.n_loading_tests", int(tests.t_column.astype(str).str.startswith("load_").sum()))
    B["ledger_loadings_counted"] = int(tests.t_column.astype(str).str.startswith("load_").sum()) > 0
    il_t = float(one(V.csv(U / "sentiment" / "panel_b.csv"), state="illiq_ar2", controls="CMKT+CMOM",
                     portfolio="Long-Short").t)
    V.put("il.ls.bonf", "passes" if abs(il_t) > float(one(lg, source="ALL").bonferroni_crit_t) else "does not pass")
    w0 = V.csv(d / "world0_calibration.csv")
    w0r = w0[w0.stat.isin(["fm3_ResLnA", "fm4_ResLnA"])]
    crit_res = float(w0r.crit_abs_t_5pct.max())
    V.put("w0.crit_res_min", fx(w0r.crit_abs_t_5pct.min(), 2))
    V.put("w0.crit_res_max", fx(crit_res, 2))
    res = B["res_rows"]
    res_ps = res[res.paper_t.abs() >= 1.96]
    V.put("sc.res_rep_w0", int(((res_ps.match == "yes") & (res_ps.us_t.abs() >= crit_res)).sum()))
    # world 0 calibrated fm3/fm4 only, full sample at the base N x T: the rows that calibration covers
    cal = res_ps[res_ps.id.isin(["FMResLnA", "FMResLnAfour"])]
    V.put("sc.res_cal_n", len(cal))
    V.put("sc.res_cal_w0", int(((cal.match == "yes") & (cal.us_t.abs() >= crit_res)).sum()))
    below = res[(res.us_t.abs() >= 1.96) & (res.us_t.abs() < crit_res)]
    V.put("sc.res_below_w0", len(below))
    V.put("sc.res_below_w0_rows", "; ".join(f"{r.metric} (t {fx(r.us_t)})" for r in below.itertuples())
          if len(below) else "none")
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
    sim_mde = one(mde, stat="fm2_lnA_C", method="sim_scaled_to_real_crit_1.96").mde
    V.put("mde.fm2.sim", fx(sim_mde, 2))
    V.put("mde.fm2.sim_ratio", fx(one(mde, stat="fm2_lnA_C", method="real_se_nw_crit_1.96").mde / sim_mde, 2))
    # per-s.d. numbers are each in their own s.d. (ours: mean monthly cs s.d. in the FM sample; paper: 2.53, pp. 13-14),
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


# ------------------------------------------------------------------------------------ reproducibility evidence
REPRO_KEYS = ["repro.status", "repro.commit", "repro.steps", "repro.rc", "repro.data_min", "repro.n_files",
              "repro.n_csv", "repro.n_identical", "repro.n_textdiff", "repro.n_numdiff", "repro.daily",
              "repro.verdict", "repro.tests", "repro.groups", "repro.n_png", "repro.n_png_ident", "repro.not_covered",
              "repro.hf_min", "repro.code_vs", "repro.cover_text"]
FC_KEYS = ["fc.status", "fc.code", "fc.head", "fc.rc", "fc.min", "fc.tests", "fc.n_man", "fc.n_canon", "fc.clean",
           "fc.hist", "fc.prev", "fc.base_n", "fc.base_maxdiff", "fc.base_numdiff", "fc.files", "fc.files_ident",
           "fc.verdict", "fc.data", "fc.home_n", "fc.home_list", "fc.home_text"]


def repro_values(V: Values, R: Path) -> None:
    """Two clean clones (src/repro_check.py): from scratch (results/repro/) and `make all` (results/repro/final_clone/)."""
    d = R / "repro"
    sj = d / "repro_summary.json"
    if not sj.exists():
        for k in REPRO_KEYS:
            V.put(k, "n/a")
    else:
        sm = V.json(sj)
        V.put("repro.status", "done")
        V.put("repro.commit", sm["clone_commit"][:7])
        V.put("repro.steps", " → ".join(x["target"] for x in sm["steps"]))
        V.put("repro.rc", "every step exited with code 0" if sm["all_steps_rc0"] else "NOT every step exited with code 0")
        data = [x for x in sm["steps"] if x["target"] == "data"]
        V.put("repro.data_min", fx(data[0]["seconds"] / 60, 0) if data else "n/a")
        hf = [x for x in sm["steps"] if x["target"] == "upgrade-hf_lambda"]
        V.put("repro.hf_min", fx(hf[0]["seconds"] / 60, 0) if hf else "n/a")
        V.put("repro.n_files", int(sm["n_files_compared"]))
        V.put("repro.verdict", sm["verdict"])
        V.put("repro.not_covered", ", ".join(f"`{u}`" for u in sm["not_covered"]) if sm["not_covered"] else "none")
        V.put("repro.cover_text", "all seven upgrades were regenerated and compared, including `upgrade-hf_lambda` "
              f"(about 5 GB of downloads, {V['repro.hf_min']} min)" if not sm["not_covered"] else
              "upgrades not regenerated by this clone, so not compared: " + V["repro.not_covered"])
        ck = V.csv(d / "repro_check.csv")
        kind = ck["kind"] if "kind" in ck else pd.Series("csv", index=ck.index).where(ck.group != "data", "parquet")
        cs = ck[kind == "csv"]                       # result CSVs only (data and figures are reported separately)
        V.put("repro.n_identical", int((cs.status == "identical").sum()))
        V.put("repro.n_textdiff", int((cs.status == "numbers_identical_text_differs").sum()))
        V.put("repro.n_numdiff", int((~cs.status.isin(["identical", "numbers_identical_text_differs"])).sum()))
        dl = ck[ck.group == "data"]
        V.put("repro.daily", f"{dl.status.iloc[0]} ({num(dl.rows_clone.iloc[0])} rows)" if len(dl) else "not compared")
        V.put("repro.n_csv", int(len(cs)))
        g = cs.groupby("group", sort=False).size()
        V.put("repro.groups", ", ".join(f"{'base run' if k == 'base' else k} {n}" for k, n in g.items()))
        pg = ck[kind == "png"]
        V.put("repro.n_png", int(len(pg)))
        V.put("repro.n_png_ident", int((pg.status == "identical").sum()))
        V.put("repro.code_vs", f"the clone ran commit `{sm['clone_commit'][:7]}`; the outputs it is compared with were "
              f"produced by commit `{V.json(R / sm['repo_base_run'] / 'run_manifest.json')['git']['sha'][:7]}`")
        tl = d / "clone_logs" / "make_tests.txt"
        m = re.findall(r"^(\d+ passed.*?) in [0-9.]+s", V.text(tl), re.M) if tl.exists() else []
        V.put("repro.tests", m[-1] if m else "not logged")

    fj = d / "final_clone" / "summary.json"
    if not fj.exists():
        for k in FC_KEYS:
            V.put(k, "n/a")
        return
    fc = V.json(fj)
    mf = V.csv(d / "final_clone" / "manifests.csv")
    V.put("fc.status", "done")
    V.put("fc.code", fc["code_commit_run"] if isinstance(fc["code_commit_run"], str) else "several commits")
    V.put("fc.head", fc["clone_head_now"])
    V.put("fc.rc", f"exit code {fc['make_all']['rc']}" if fc["make_all"] else "not logged")
    V.put("fc.min", fx(fc["make_all"]["seconds"] / 60, 0) if fc["make_all"] else "n/a")
    V.put("fc.tests", fc["tests_summary"] or "not logged")
    V.put("fc.n_man", fc["n_manifests"])
    V.put("fc.n_canon", fc["n_canonical"])
    V.put("fc.clean", "all record that commit with `dirty = false`" if fc["canonical_all_same_commit_clean"]
          else "NOT all record one commit with `dirty = false`")
    V.put("fc.hist", ", ".join(f"`{h['run_id']}` (`{h['git_sha']}`, {'dirty' if h['dirty'] else 'clean'})"
                               for h in fc["historical_runs"]) or "none")
    V.put("fc.prev", fc["base_compared_with"] or "none")
    V.put("fc.base_n", fc["base_n_csv"])
    V.put("fc.base_maxdiff", fx(fc["base_max_abs_diff"], 0) if fc["base_max_abs_diff"] is not None else "n/a")
    V.put("fc.base_numdiff", fc["base_n_not_identical"])
    V.put("fc.files", fc["results_files_in_clone"])
    V.put("fc.files_ident", fc["results_files_identical_to_repo"])
    V.put("fc.verdict", fc["verdict"])
    V.put("fc.data", "linked to the cached downloads of the main working copy" if fc["data_dir_was_link"]
          else "a directory of the clone")
    home = mf[(mf.role == "canonical") & (mf.n_home_paths > 0)]
    V.put("fc.home_n", int(len(home)))
    V.put("fc.home_list", ", ".join(f"`{Path(p).parent.name}`" for p in home.manifest) or "none")
    V.put("fc.home_text", (
        f"{len(home)} of the {fc['n_canonical']} canonical manifests ({V['fc.home_list']}) hold relative paths that "
        "run through the `data/` link into the main working copy (`../…/Users/…`); `tests/test_manifests.py` rejects "
        "them (a failure for the cheap upgrades, an xfail for the base run and `hf_lambda`), so `make tests` on the "
        "committed results fails until these manifests are regenerated without the link"
        if len(home) else f"none of the {fc['n_canonical']} canonical manifests holds an absolute or home-directory path"))


def tests_log_values(V: Values, R: Path) -> None:
    """Latest saved pytest log (results/tests_log/) and whether it was run on the code state the docs are built from."""
    logs = sorted((R / "tests_log").glob("pytest_*.txt"))
    if not logs:
        for k in ["tests.log", "tests.summary", "tests.date", "tests.code_match"]:
            V.put(k, "n/a")
        return
    txt = V.text(logs[-1])
    V.put("tests.log", str(logs[-1].relative_to(ROOT)))
    m = re.findall(r"^(\d+ passed[^\n]*?) in [0-9.]+s", txt, re.M)
    V.put("tests.summary", m[-1] if m else "no pytest summary line")
    d = re.search(r"^date: (\S+)", txt, re.M)
    V.put("tests.date", d.group(1) if d else "n/a")
    c = re.search(r"^code state: (\S+)", txt, re.M)
    V.put("tests.code_match", ("matches" if c.group(1) == code_state_hash() else "does NOT match") if c
          else "cannot be compared with (the log has no code-state line)")


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
        for k in ["git.refine", "git.firstrun", "git.firstrun.sha", "git.firstrun.cfg8", "git.prereg.cfgmatch",
                  "fr.fm3res", "fr.res_tmax", "fr.t4_tmax"]:
            V.put(k, "n/a")
        return
    _, mp, first = min(mans, key=lambda x: x[0])
    V.json(mp)
    fsha = first["git"]["sha"]
    refine = git("log", "--format=%h", f"{B['prereg']}..{fsha}", "--", "DEVIATIONS.md", "config.yaml").split()
    V.put("git.refine", ", ".join(f"`{c}`" for c in reversed(refine)).strip("`") if refine else "none")
    V.put("git.firstrun", first["run_id"])
    V.put("git.firstrun.sha", fsha[:7] + (" (dirty)" if first["git"].get("dirty") else ""))
    V.put("git.firstrun.cfg8", first["config_sha256"][:8])
    pre_cfg = hashlib.sha256(subprocess.run(["git", "show", f"{B['prereg']}:config.yaml"], cwd=ROOT,
                                            capture_output=True).stdout).hexdigest()
    V.put("git.prereg.cfgmatch", "matches" if pre_cfg == first["config_sha256"] else "does NOT match")
    # the pre-registered first run (before the J1 data audit): its T4 Res.lnA slopes
    t4 = V.csv(mp.parent / "T4_fm.csv")
    r = one(t4, model="(3)", var="ResLnA")
    V.put("fr.fm3res", ct(r.coef, r.t))
    V.put("fr.res_tmax", fx(t4[t4["var"] == "ResLnA"].t.abs().max(), 2))
    V.put("fr.t4_tmax", fx(t4.t.abs().max(), 2))


# ------------------------------------------------------------------------------------ blocks
def blocks(V: Values, B: dict, R: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    rd: Path = B["dir"]
    up = B["upgrade_manifests"]

    out["provenance"] = (
        f"_The numbers in this file are generated by `src/build_docs.py` from CSV files: base run `{V['base.run']}` "
        f"(git `{V['base.git']}`, config sha256 `{V['base.cfg8']}…`), simulations `results/simulations/` "
        f"(`{V['sim.run']}`, config `{V['sim.cfg8']}…`, git `{V['sim.git']}`), upgrades "
        + ", ".join(f"`{k}` → `{m['run_id']}`" for k, m in sorted(up.items()))
        + ". Refresh with `make docs`. Values between `<!--v:...-->` markers are never edited by hand. "
        "`(dirty)` means the result was produced from a working copy with uncommitted changes. The manifest field "
        f"`code_sha256` (sha256 of `src/*.py`) records that code state; for the base run (`{V['base.code8']}…`) it "
        f"{V['base.code_commit']}, {V['base.code_checkout']}"
        + (f"; `make all` on a clean clone of `{V['fc.code']}` wrote these outputs ({V['fc.rc']}, "
           "`results/repro/final_clone/`)" if V["fc.status"] == "done" else "")
        + (f"; a from-scratch clone of `{V['repro.commit']}` (every download redone) reproduces them numerically "
           f"(verdict “{V['repro.verdict']}”, `results/repro/`)" if V["repro.status"] == "done" else "")
        + " (see “Reproducibility”)._")

    # paper vs us: the generated report itself (sections Key findings + Paper vs us)
    md = V.text(rd / "paper_vs_us.md")
    m = re.search(r"## Key findings\n(.*?)\n## Robustness", md, re.S)
    if not m:
        raise LookupError("paper_vs_us.md: sections not found")
    body = m.group(1).replace("\n## Paper vs us\n", "\n#### Paper vs us\n")
    hf_win = f"Binance {V['hf.first']}…{V['hf.last']}; paper T9 1983–2012"
    extra = md_table(
        ["Metric", "Paper", "Us", "Market and window", "Source"],
        [["corr(AT, AT_C)", V["mx.corrAT.paper"], V["mx.corrAT"], f"MOEX TQBR {V['mx.first']}…{V['mx.last']}",
          "`results/upgrades/moex/paper_vs_moex.csv`"],
         ["corr(A, λ)", V["hf.corr.A.paper"], V["hf.corr.A"], hf_win, "`results/upgrades/hf_lambda/T9A_corr.csv`"],
         ["corr(A_C, λ)", V["hf.corr.A_C.paper"], V["hf.corr.A_C"], hf_win, "`results/upgrades/hf_lambda/T9A_corr.csv`"],
         ["corr(AT_C, λ)", V["hf.corr.ATC.paper"], "not computed",
          "Binance has no token supply (no AT_C); MOEX has no λ", "`src/paper_numbers.yaml` (paper value only)"],
         ["FM ln λ, T9B model (1)", V["hf.b1.paper"], V["hf.b1"], hf_win, "`results/upgrades/hf_lambda/T9B_fm.csv`"],
         ["FM ln λ, T9B model (2) with ln A_C", V["hf.b2lam.paper"], V["hf.b2lam"], hf_win,
          "`results/upgrades/hf_lambda/T9B_fm.csv`"]])
    notes = "\n".join([
        f"- **Sign agreement.** {V['sc.samesign']} of {V['sc.ls_n']} coefficient rows have the paper's sign. "
        f"{V['sc.yes_oppsign']} of the {V['sc.yes']} 'yes' rows have the opposite sign but are insignificant in "
        f"both samples, which the match rule counts as a match ({V['sc.yes_oppsign_rows']}). The rows overlap "
        "heavily (the same few signals in several specifications), so sign agreement is weak evidence.",
        f"- **Res. ln A against the world-0 threshold.** The paper's Res. ln A coefficient is significant in "
        f"{V['sc.res_paper_sig']} of our {V['sc.res_n']} Res. ln A rows. We repeat that significance ('yes') in "
        f"{V['sc.res_rep196']} of them at |t| ≥ 1.96 and in {V['sc.res_rep_w0']} at the world-0 empirical 5% "
        f"critical |t| ({V['w0.crit_res_min']}–{V['w0.crit_res_max']}, `results/upgrades/robustness/world0_calibration.csv`); "
        "none pass the HLZ bar of 3. Rows with 1.96 ≤ |t| below the world-0 critical value: "
        f"{V['sc.res_below_w0_rows']}. World 0 calibrated this threshold only for the full-sample models (3) and "
        f"(4) at the base N×T ({V['sc.res_cal_w0']} of those {V['sc.res_cal_n']} rows pass it); for the half-sample, "
        "Table 6 and raw-return rows the same threshold is only an approximation.",
    ])
    out["paper_vs_us"] = (f"_Below is `{V['base.dir']}/paper_vs_us.md` verbatim (generated by `src/report.py`), "
                          "sections “Key findings” and “Paper vs us”._\n\n#### Key findings\n" + body.strip()
                          + "\n\n#### Notes on the table (added by `src/build_docs.py`)\n\n" + notes
                          + "\n\n#### From the upgrades (rows of task §5 that the base run cannot produce)\n\n"
                          "Each row has its own market and window; t in brackets (Newey–West, 6 lags). λ is our "
                          "5-minute-bar λ with the trade sign taken from the bar's aggregate taker-buy volume (no "
                          "Lee–Ready); the paper's λ is from TAQ with Lee–Ready.\n\n"
                          + extra)

    out["figure_main"] = (f"![Quintiles of A / A_C / Res.A: paper and us]({V['base.dir']}/fig_quintiles.png)\n\n"
                          f"_File: `{V['base.dir']}/fig_quintiles.png` (generated by `src/figures.py`)._")

    # audit list
    ev = B["audit_events"]
    rows = []
    for r in ev.itertuples():
        f = "" if pd.isna(r.factor) else f"{float(r.factor):.4g}"
        rows.append([r.symbol.replace("USDT", ""), r.date, r.type, f, r.action])
    out["audit"] = md_table(["coin", "date", "type", "factor (redenomination) / reopen ÷ prev. close", "treatment"],
                            rows)

    # simulations
    ws = B["sim_ws"]

    def srow(w, stat, col, expect):
        r = one(ws, world=w, stat=stat)
        return [w, stat, expect, pc(r[col], 1), fx(r.mean_t, 2), int(r.n_reps)]
    srows = [
        srow("V", "fm2_lnA_C", "share_t_gt_1.96", "priced: high share of t > 1.96"),
        srow("V", "hl_A_C", "share_t_gt_1.96", "priced"),
        srow("V", "fm3_ResLnA", "share_abs_t_gt_1.96", "not priced: share of |t| > 1.96 near 5%"),
        srow("V", "hl_ResA", "share_abs_t_gt_1.96", "not priced"),
        srow("P", "fm3_ResLnA", "share_t_gt_1.96", "priced"),
        srow("P", "hl_ResA", "share_t_gt_1.96", "priced (the sort is weaker than FM)"),
        srow("P", "fm2_lnA_C", "share_abs_t_gt_1.96", "ideally not priced"),
        srow("0", "fm2_lnA_C", "share_abs_t_gt_1.96", "false positives near 5%"),
        srow("0", "fm3_ResLnA", "share_abs_t_gt_1.96", "false positives near 5%"),
        srow("0", "fm3_ResLnA_nw0", "share_abs_t_gt_1.96", "same, without NW lags"),
        srow("0", "hl_A_C", "share_abs_t_gt_1.96", "false positives near 5%"),
    ]
    sh = ["world", "statistic", "expectation", "share significant", "mean t", "replications"]
    t = md_table(sh, srows)
    B.setdefault("tables", {})["sims"] = (sh, [[str(c) for c in r] for r in srows])
    chk = B["sim_checks"]
    failed = chk[chk["pass"].astype(str).str.lower() != "true"]
    fl = "\n".join(f"- failed: world {r.world}, {r.criterion} = {fx(r.value, 3)} (rule `{r.rule}`)"
                   for r in failed.itertuples())
    pw = B["sim_power"]
    prow = []
    for prem in sorted(pw.premium.unique()):
        a = one(pw, premium=prem, stat="fm2_lnA_C")
        b = one(pw, premium=prem, stat="hl_A_C")
        prow.append([fx(100 * prem, 2), pc(a["share_t_gt_1.96"]), V[f"pow.{fx(100 * prem, 2)}.fm2.ci"],
                     pc(b["share_t_gt_1.96"]), V[f"pow.{fx(100 * prem, 2)}.hlAC.ci"], int(a.n_reps)])
    pt = md_table(["premium, %/mo per 1 s.d.", "power FM ln A_C", "95% CI", "power H-L A_C", "95% CI",
                   "replications"], prow)
    out["simulations"] = (t + f"\n\nCriteria in `sim_checks.csv`: {V['sim.checks.pass']} of "
                          f"{V['sim.checks.n']} passed ({V['sim.checks.loose_n']} of them use the loose rule "
                          f"“share {V['sim.checks.loose_rule']}” for unpriced signals; with ≤ 5% instead, "
                          f"{V['sim.checks.pass_strict5']} of {V['sim.checks.n']}). {V['git.sim_rules']}\n" + fl + "\n\nPower in world V (`sim_power.csv`; "
                          f"{V['pow.reps']} replications per point, so each share is coarse, see the 95% CI "
                          "of the share of |t| > 1.96):\n\n" + pt)

    # upgrades table
    U = "results/upgrades"
    urows = [
        ["1. Amihud & Noh (2021): the missing term",
         "Is DIF = ln A − ln|Ret| − ln A_C priced beyond A_C? Who is right?",
         f"COV < 0 in {V['an.covneg']} of coin-months; DIF col. 3: {V['an.V1m']} (monthly), {V['an.V1y']} (12-month) "
         f"vs A&N {V['an.p.col3']}; MDE80 of the monthly DIF {V['pvu.ANDIF.mde']} in our units; fDIF (12-month) {V['an.V4y']} vs A&N {V['an.p.fdif']}; Res.lnA {V['an.V3m']} → "
         f"{V['an.V5m']} when ln|Ret| is added to model (5) (almost mechanical: R² of Res.lnA on ln|Ret| + DIF = {V['an.r2_res_dif']})",
         "the data do not separate LS from A&N. DIF is insignificant; our coefficients are on a different scale "
         f"from the papers' (|ours / paper| {V['an.scale_min']}–{V['an.scale_max']} in the key-regression rows of "
         f"`amihud_noh/SUMMARY.md`, e.g. {V['an.scale_eg']}), so the MDE cannot be set "
         "against A&N's coefficient and our power against an A&N-sized effect is unknown: insignificance does not "
         "show that DIF is unpriced. The only result in A&N's direction in the main "
         f"specification is fDIF on 12-month measures (below HLZ); on raw returns the monthly DIF in V2 is "
         f"{V['an.ret.V2m']} and fDIF {V['an.ret.V4y']}. The verdict of our rule (not pre-registered: code and results were committed together) changes with the minimum number "
         f"of coins per month ({V['an.sens.levels']}: {V['an.sens.ret_adj']}; fDIF t {V['an.sens.fdif_t']}; this "
         f"sensitivity was added during checking, post hoc); |t| ≥ 3: {V['an.sens.n3']}",
         f"`{U}/amihud_noh/`"],
        ["2. λ from 5-minute bars (Tables 9/10)",
         "Does price impact λ (sign from the bars' aggregate taker volume in 5-minute klines) explain the premium?",
         f"{V['hf.months']} months ({V['hf.first']}…{V['hf.last']}); bar λ against trade-level λ from aggTrades: "
         f"corr of ln λ {V['hf.val.corr']}, Spearman {V['hf.val.spearman']}, {V['hf.val.n']} coin-months in "
         f"{V['hf.val.n_months']} months ({V['hf.val.months']}); bar λ is about {V['hf.val.ratio']}× the trade-level "
         f"λ in level, so only logs and ranks are comparable. corr(λ, A) {V['hf.corr.A']} (paper "
         f"{V['hf.corr.A.paper']}); FM ln λ (1) {V['hf.b1']} (paper {V['hf.b1.paper']}), (2) {V['hf.b2lam']} (paper "
         f"{V['hf.b2lam.paper']}); post hoc diagnostics: without exit months (2) {V['hf.exit.b2lam']}, with IVOL (1) "
         f"{V['hf.ivol.b1']}, with lnVOL (1) {V['hf.vol.b1']}; ln A_C in this window without λ "
         f"{V['hf.b0']}; T10 (3): cost {V['hf.t10.c3']} (paper {V['hf.t10.c3.paper']}), non-cost {V['hf.t10.n3']} "
         f"(paper {V['hf.t10.n3.paper']})",
         f"ln λ is negative over one window of {V['hf.months']} monthly cross-sections; the size of NW(6) at "
         "this T is not calibrated (world 0 uses the base N×T). It diverges from the paper in Table 9 model (1) and in Table 10 "
         "(our cost part is significantly negative where the paper's is zero; in T10 model (3) our non-cost part "
         f"is insignificant, {V['hf.t10.n3']}, where the paper's is significant, {V['hf.t10.n3.paper']}; alone, in "
         f"model (2), it is {V['hf.t10.n2']}, the paper's sign, at 1.96 only, against {V['hf.t10.n2.paper']}). In model (2) the signs agree (ln λ {V['hf.b2lam']} vs "
         f"{V['hf.b2lam.paper']}; ln A_C {V['hf.b2ac']} vs {V['hf.b2ac.paper']}). In this window the A_C premium is "
         f"absent (ln A_C without λ {V['hf.b0']}: negative point estimate, insignificant), so the λ result may be a "
         "window effect; Table 9 is not reproduced",
         f"`{U}/hf_lambda/`"],
        ["3. EDGE spread and costs", "Is the A_C premium achievable after costs? Does the spread explain A?",
         f"H-L A_C gross {V['ed.gross']}, net {V['ed.net']}, with a {V['ed.fee']} fee {V['ed.netfee']}; break-even "
         f"cost {V['ed.breakeven']} per side; max |t| across the EDGE models {V['ed.maxabs_t']}",
         "daily EDGE does not measure the spread (its weight sits on the open − previous close gap); the premium is "
         "insignificant even before costs, and the break-even cost is computed from that insignificant point "
         "estimate (see Limitations)",
         f"`{U}/edge_spread/`"],
        ["4. Sentiment (Table 13)", "Is the premium larger after high Fear & Greed (mispricing)?",
         f"L−S {V['se.ls']} (paper {V['se.ls.paper']}); short {V['se.short']} (paper {V['se.short.paper']}); long "
         f"{V['se.long']} (paper {V['se.long.paper']}); FM slope of ln A_C on sentiment {V['se.fm']}; with "
         f"CMKT(t−1), post hoc, L−S {V['se.post.ls']}",
         f"not confirmed: the short leg and L−S have the paper's sign, the long leg does not; {V['se.var_196']} of "
         f"{V['se.var_n']} L−S portfolio variants not labelled post hoc pass 1.96. The FM slope {V['se.fm']} is borderline "
         "at 1.96 and below HLZ (world 0 did not calibrate this time-series statistic)", f"`{U}/sentiment/`"],
        ["5. Market illiquidity (Table 12)", "Is the premium larger when the market is illiquid?",
         f"L−S {V['il.ls']} (paper {V['il.ls.paper']}); log state {V['il.log']}; with CMKT(t−1), post hoc, "
         f"{V['il.post.ls']}; median split {V['il.split']}",
         f"the paper's sign; {V['il.var_3']} of {V['il.var_n']} variants not labelled post hoc pass 3, and the main t "
         f"{V['il.ls.bonf']} Bonferroni over the whole ledger (|t| {V['lg.all.crit']}). The AR(2) state is fitted on the full sample "
         "(look-ahead), and because A contains |r| our state mixes illiquidity with market volatility, so we do not "
         "claim the paper's “against liquidity” interpretation", f"`{U}/sentiment/`"],
        ["6. MOEX TQBR (second market, AT/AT_C)", "Does the decomposition hold on Russian stocks with turnover?",
         f"FM ln A_C (2) {V['mx.fm2']} (paper {V['mx.fm2.paper']}); Res.lnA (4) {V['mx.fm4res']} (paper "
         f"{V['mx.fm4res.paper']}); raw returns ln A_C (2) {V['mx.raw.fm2ac']}, (4) {V['mx.raw.fm4ac']}; H-L A_C {V['mx.hlAC']}, alpha "
         f"{V['mx.hlACalpha']}, SMB loading {V['mx.smb']}; H-L Res.A {V['mx.hlResA']} (paper {V['mx.hlResA.paper']})",
         f"partial: the positive ln A_C slope repeats at 1.96 (not HLZ; insignificant on raw returns over all their "
         f"months, but over the months of the main FM, post hoc, model (4) gives {V['mx.rawsame.fm4ac']}), the paper's "
         f"negative Res.lnA does not; the insignificant sort spread loads on SMB; post hoc: {V['mx.topyear_share']} of the "
         f"cumulative H-L comes from {V['mx.topyear']}", f"`{U}/moex/`"],
        ["7. US after 2012 (OpenAP)", "Is the Illiquidity/DolVol premium alive after the paper's sample? Table A.1",
         f"raw L−S Illiquidity {V['op.ill.pre']} → {V['op.ill.post']}; DolVol {V['op.dv.pre']} → {V['op.dv.post']}; "
         f"FF3+UMD alpha of Illiquidity {V['op.pre.ff4']} → {V['op.post.ff4']}; SMB loading {V['op.pre.smb']} → "
         f"{V['op.post.smb']}; alpha of Illiquidity on DolVol {V['op.pre.one']} (paper {V['op.paper.one']})",
         f"the raw L−S fell after 2012 (difference {V['op.ill.decay']}; t across NW lags {V['op.ill.decay_lags']}: "
         f"{V['op.ill.decay_ts']}); its FF3+UMD alpha is insignificant in both periods and the SMB loading fell, "
         "consistent with the drop coming from size exposure (not tested formally: there is no test of the alpha "
         "difference). The OpenAP portfolio (NYSE only, price > 5, 12-month measure, annual rebalancing) is not the "
         "paper's design (NYSE/AMEX, monthly A at t−2, monthly rebalancing)", f"`{U}/openap_us/`"],
        ["8. Robustness and overfitting (depth item, task §8/§13)",
         "How well do the conclusions hold: bootstrap, years, power, multiplicity",
         f"bootstrap CI excludes zero robustly: {V['rb.robust']} of {V['rb.n']} (both Res.lnA); without 2021 H-L A_C "
         f"{V['x21.hlAC']}; MDE80 of FM ln A_C {V['mde.fm2.slope']} per unit of ln A_C vs the paper's slope "
         f"{V['mde.fm2.paper_slope']}",
         f"the null is uninformative (MDE {V['mde.fm2.slope_ratio']}× the paper's slope per unit of ln A_C, "
         f"{V['mde.fm2.ratio']}× per 1 s.d.); in the base family nothing survives Bonferroni/BH",
         f"`{U}/robustness/`"],
    ]
    out["upgrades"] = md_table(["upgrade", "question", "result (t: Newey–West, 6 lags)", "verdict", "folder"], urows)

    # overfitting
    lg_src = f"base {V['lg.base.3']}, " + V["lg.all.3_by_source"]
    out["overfitting"] = "\n".join([
        f"- **Pre-registration.** The specification (`config.yaml`, `DEVIATIONS.md` D1–D11) was committed in "
        f"`{V['git.prereg']}`; refinements of `DEVIATIONS.md` (D7 rewritten, D12 and D13 added) in `{V['git.refine']}`, "
        f"also before the first real-data run (`{V['git.firstrun']}`, git `{V['git.firstrun.sha']}`). The "
        f"`config.yaml` of the first run {V['git.prereg.cfgmatch']} the pre-registered one (sha256 "
        f"`{V['git.firstrun.cfg8']}…`). Changes after the first run are in the journal (J1 data audit, "
        "J2 winsorization), each with a reason. J1 matters for the results: in the pre-registered first run, which "
        f"still had the unadjusted COCOS redenomination, no FM slope in T4 reached |t| = 1.96 (largest |t| "
        f"{V['fr.t4_tmax']}; Res.lnA in model (3) {V['fr.fm3res']}, `{V['git.firstrun']}/T4_fm.csv`). The audit "
        "rules (thresholds and the gap rule) were designed after that run, so the Res.lnA significance depends on "
        "a data fix made after the first run.",
        f"- **Test ledger** (`results/upgrades/robustness/ledger_summary.csv`). Base family: {V['lg.base.n']} unique "
        f"tests, |t| > 1.96 in {V['lg.base.196']} (with all effects zero we would expect {V['lg.base.exp']} at the "
        f"nominal 5% and about {V['w0.exp_base']} at the world-0 size of NW(6)), |t| > 3 in {V['lg.base.3']}; after "
        f"Bonferroni (|t| threshold {V['lg.base.crit']}) {V['lg.base.bonf']}, after BH {V['lg.base.bh']}. Base and "
        f"upgrades together: {V['lg.all.n']} unique tests ({V['lg.all.rows']} ledger rows with role “test”), |t| > 3 in {V['lg.all.3']} (by source: {lg_src}), "
        f"Bonferroni (|t| threshold {V['lg.all.crit']}) {V['lg.all.bonf']}, BH {V['lg.all.bh']}. "
        + (f"These counts overstate the number of distinct findings: the ledger also labels "
           f"{V['lg.all.n_loading_tests']} factor-loading rows, the ln ME control and identical estimates under "
           "different NW lags as tests. " if B["ledger_loadings_counted"] else "")
        + f"By kind, the Bonferroni survivors are: {V['lg.all.bonf_by_kind']}. "
        + (f"Of these, {V['lg.all.bonf_loadings']} are MOEX factor loadings (not hypothesis tests); "
           if int(V['lg.all.bonf_loadings']) else "")
        + f"the {V['lg.all.bonf_hf_n']} hf_lambda rows all come from the one {V['hf.months']}-month λ window "
        f"({V['lg.all.bonf_hf']}); the {V['lg.all.bonf_sent_n']} sentiment row(s): {V['lg.all.bonf_sent']} (the main "
        f"market-illiquidity specification uses controls {V['il.main_controls']}).",
        f"- **Test size.** In world 0, NW(6) on our N×T gives false |t| > 1.96 in {V['w0.fpr_min']}–{V['w0.fpr_max']} "
        f"of runs (|t| > 3 in at most {V['w0.fpr3_max']}); the empirical 5% critical |t| is {V['w0.crit_min']}–"
        f"{V['w0.crit_max']}. Simulation criteria passed: {V['sim.checks.pass']} of {V['sim.checks.n']}; failed: "
        f"{V['sim.checks.failed']}. {V['sim.checks.loose_n']} of the passed criteria use the loose rule "
        f"“share {V['sim.checks.loose_rule']}” for signals that should be unpriced (world P falsely prices ln A_C in "
        f"{V['sim.checks.loose_max']} of runs); with a ≤ 5% rule for those, {V['sim.checks.pass_strict5']} of "
        f"{V['sim.checks.n']} would pass. The P0 item “about 5% in world 0” is therefore not met, which is why we show the "
        "world-0 empirical threshold and the HLZ bar of 3 next to 1.96. This calibration covers the base N×T only "
        f"(not the {V['hf.months']}-month λ window or the overlapping 12-month A&N measures).",
        f"- **Block bootstrap over months** ({V['rb.block']}-month blocks, {V['rb.draws']} draws, plus {V['rb.seeds']} "
        f"seeds): the CI excludes zero for {V['rb.excl']} of {V['rb.n']} statistics, robustly across seeds for "
        f"{V['rb.robust']} (FM Res.lnA (3) {V['rb.ci.fm3_ResLnA']}, (4) {V['rb.ci.fm4_ResLnA']}); FM ln A_C (4) is "
        f"borderline ({V['rb.seedshare.fm4_lnA_C']} seeds exclude zero).",
        f"- **Placebo** (`placebo.csv`): shuffling the measures within a month ({V['plc.shuffle_reps']} permutations, "
        f"so the shares are coarse): mean t across permutations at most {V['plc.shuffle_maxabs_meant']} in absolute "
        f"value, s.d. of t across permutations {V['plc.shuffle_sd_t_min']}–{V['plc.shuffle_sd_t_max']}, share of "
        f"|t| > 1.96 at most {V['plc.shuffle_share196_max']}. A leak placebo (lag 0 instead of t−2) gives |t| "
        f"{V['plc.leak_minabs_t']}–{V['plc.leak_maxabs_t']}. Lag 0 is a contemporaneous measure, so part of that |t| "
        "is the mechanical return–volume link; the future-lag (t+1) placebo of task §13.1 was not run.",
        f"- **Stability.** Without 2021, H-L A_C {V['x21.hlAC']}, FM ln A_C {V['x21.fm2']}, Res.lnA (3) "
        f"{V['x21.fm3']}; leaving out one year, the t of H-L A_C ranges over {V['loyo.hlAC']}. Winsorized returns, "
        f"raw returns, weighting, sub-periods and without the most influential month: {V['rob.same']} of "
        f"{V['rob.n']} variants keep the base conclusion (`robustness_vs_base.csv`; same sign and same side of 1.96), "
        f"including {V['rob.posthoc_n']} post hoc variants without the most influential month (chosen from the data "
        "and added after the first real-data run). "
        f"{V['rob.same_insig']} of these {V['rob.same']} are insignificant estimates that stay insignificant; "
        f"{V['rob.same_sig']} of the {V['rob.sig_n']} variants of the significant statistics ({V['rob.sig_stats']}) "
        "stay significant.",
        f"- **Power.** MDE80 for H-L A_C is {V['mde.hlAC']}%/mo against the paper's premium of {V['mde.hlAC.paper']}. "
        f"For FM ln A_C: {V['mde.fm2.slope']} per unit of ln A_C against the paper's slope "
        f"{V['mde.fm2.paper_slope']} ({V['mde.fm2.slope_ratio']}× larger); per 1 s.d., {V['mde.fm2']} (our s.d. "
        f"{V['mde.fm2.sd']}) against {V['mde.fm2.paper']} for the paper (its s.d. {V['mde.fm2.paper_sd']}, "
        f"pp. 13–14), i.e. {V['mde.fm2.ratio']}×, or {V['mde.fm2.paper_oursd']} if the paper's slope is multiplied by "
        f"our s.d. The real-data MDE is {V['mde.fm2.sim_ratio']}× the simulation MDE, so the simulations understate "
        "the noise. Insignificance here does not show that a premium of the paper's size is absent.",
        "- **Survivorship and look-ahead.** The universe includes USDT pairs that were later delisted (they are in "
        "the `data.binance.vision` listing); whether that archive holds every pair ever traded was not checked, the "
        "number of delisted pairs is not recorded in `results/`, and delisting returns are missing (D10). The signal "
        "is taken from t−2: `tests/test_alignment.py` recomputes A, A_C, returns, momentum, CMKT, beta, ret_adj and "
        "IVOL from raw prices for 40 random rows of a synthetic panel, and its mutation tests (signal lag 1 or 0) "
        "show that the check flags a shifted signal. It does not cover Res.A/Res.lnA, |Ret|, DIF, the 12-month "
        "A&N measures or the separate panels of the upgrades (`hf_lambda`, `moex`, `sentiment`). Latest saved test "
        f"log: `{V['tests.log']}` ({V['tests.summary']}; its code-state hash {V['tests.code_match']} the code these "
        "docs were built from). Uncommitted work: see “What was not done”.",
        "- **Post hoc variants** (added after we saw the main result; not pre-registered) are labelled in the "
        "`SUMMARY.md` of each upgrade and are never the main specification. In the CSVs, "
        "`hf_lambda/variants.csv` and `sentiment/panel_b.csv` carry a post hoc column, and "
        f"`{V['base.dir']}/robustness_vs_base.csv` marks its {V['rob.posthoc_n']} post hoc rows in the variant label.",
    ])

    # runtime and disk
    rrows = [["`make run` (base run)", fsec(B["manifest"]["timings_sec"]["total"]), f"`{V['base.dir']}/run_manifest.json`"],
             ["`make sims` (simulations, parallel)", V["sim.wall"], "`results/simulations/sim_manifest.json`"]]
    rep = rd / "report_manifest.json"
    if rep.exists():
        rm = V.json(rep)
        rrows.append(["`make report`", fsec(sum(rm["timings_sec"].values())), f"`{V['base.dir']}/report_manifest.json`"])
    for k, m in sorted(up.items()):
        ts = m.get("timings_sec", {})
        tot = ts.get("total", sum(v for v in ts.values() if isinstance(v, (int, float))))
        rrows.append([f"`make upgrade-{k}`", fsec(tot), f"`results/upgrades/{k}/run_manifest.json`"])
    rt = md_table(["stage", "seconds (last run, data cache present)", "source"], rrows)
    raw = ROOT / "data" / "raw"
    drows = []
    if raw.exists():
        for p in sorted(raw.iterdir()):
            if p.is_dir():
                size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                drows.append([f"`data/raw/{p.name}/`", fx(size / 1e6, 1)])
    dt = (md_table(["folder", "MB (10^6 bytes) on the author's machine"], drows) if drows
          else "_data/raw/ is absent on this machine._")
    out["runtime"] = (rt + "\n\nDownload time is not written to the manifests because it depends on the network. "
                      "Size of the raw data (measured when the docs are built):\n\n" + dt)

    # manifest fields
    man = B["manifest"]
    out["manifest_fields"] = ("Fields of the base run's `run_manifest.json`: " + ", ".join(f"`{k}`" for k in man)
                              + ".")

    # slide 8 compact table
    pvu = B["pvu"].set_index("id")
    sel = ["corrAAC", "QratioAC", "corrAabs", "HLArawCoef", "HLACrawCoef", "HLACalphaCoef", "HLResArawCoef",
           "FMlnA", "FMlnAC", "FMResLnA", "FMResLnAfour", "IMLAalpha"]
    srows = [[pvu.loc[i, "metric"], V[f"pvu.{i}.paper"], V[f"pvu.{i}"], pvu.loc[i, "match"]] for i in sel]
    T = B.setdefault("tables", {})
    T["slide8"] = (["metric", "paper (NYSE/AMEX 1964–2012)", "us (Binance)", "match"], srows)

    # slide 9: lambda (paper: Table 9 is NYSE/AMEX 1983-2012; Table 10's caption says 1964-2012, but it reports
    # 358 months and needs λ, so it is the same 1983-2012 λ sample)
    T["slide9"] = (["", "paper (T9, T10: λ sample 1983–2012, 358 mo.)",
                    f"us (Binance {V['hf.first']}…{V['hf.last']}, 5-min bars)"], [
        ["corr(λ, A)", V["hf.corr.A.paper"], V["hf.corr.A"]],
        ["corr(λ, A_C)", V["hf.corr.A_C.paper"], V["hf.corr.A_C"]],
        ["corr(λ, Res.A)", V["hf.corr.ResA.paper"], V["hf.corr.ResA"]],
        ["FM ln λ, model (1)", V["hf.b1.paper"], V["hf.b1"]],
        ["FM ln λ with ln A_C, model (2)", V["hf.b2lam.paper"], V["hf.b2lam"]],
        ["FM ln A_C with ln λ, model (2)", V["hf.b2ac.paper"], V["hf.b2ac"]],
        ["T10: cost part, model (3)", V["hf.t10.c3.paper"], V["hf.t10.c3"]],
        ["T10: non-cost part, model (3)", V["hf.t10.n3.paper"], V["hf.t10.n3"]],
    ])

    # slide 10: mispricing. Paper columns of Tables 11-13 are turnover sorts (AT_C) with Baker-Wurgler /
    # Pastor-Stambaugh states; ours are A_C with Fear & Greed / AR(2) of EW A: only sign and t compare
    T["slide10"] = (["test", "paper (AT_C; BW / PS)", "us (A_C; Fear & Greed / AR(2) of A)"], [
        ["Sentiment → L−S (Table 13)", V["se.ls.paper"], V["se.ls"]],
        ["Sentiment → short leg", V["se.short.paper"], V["se.short"]],
        ["Sentiment → FM slope of ln A_C", "—", V["se.fm"]],
        ["Market illiquidity → L−S (Table 12)", V["il.ls.paper"], V["il.ls"]],
        ["same, + CMKT(t−1), post hoc", "—", V["il.post.ls"]],
        ["January slope (Table 11 (1); paper: ln AT_C)", V["t11.jan_atc.paper"],
         f"January {V['t7.jan']}, n {V['t7.jan.n']}"],
        ["Other months", V["t11.rest_atc.paper"], f"{V['t7.rest']}, n {V['t7.rest.n']}"],
    ])
    for k, (h, r) in T.items():
        out[k] = md_table(h, r)
    out["slide10"] += ("\n\n_Only sign and t are comparable, not scale: the paper sorts on turnover (AT_C) and "
                       "uses the Baker–Wurgler / Pástor–Stambaugh indices; we use A_C, Fear & Greed and the AR(2) "
                       "residual of the EW mean of A._")

    # team and repository (README header and title slide): from slides/meta.yaml, filled by the team
    meta = B.get("meta", {})
    team = ", ".join(meta.get("team") or [])
    repo = meta.get("repo_url") or ""
    out["team"] = (f"**Team:** {team or '_not filled in: `slides/meta.yaml` → team_'}  \n"
                   f"**Repository:** {repo or '_not filled in: `slides/meta.yaml` → repo_url_'}")
    names = meta.get("team") or []
    roles = deck_roles(V.text(ROOT / "slides" / "deck.tex"))
    lines = []
    for n in names:
        sp = ", ".join(roles["speaker"].get(n, [])) or "no main slide"
        qa = ", ".join(roles["owner"].get(n, []))
        lines.append(f"- {n}: speaks on slides {sp}" + (f"; Q&A owner of backup {qa}" if qa else ""))
    unknown = sorted((set(roles["speaker"]) | set(roles["owner"])) - set(names))
    if unknown:
        raise ValueError(f"deck.tex names people who are not in slides/meta.yaml: {unknown}")
    out["team_roles"] = ("\n".join(lines) + "\n\n_Read from the `\\speaker{}` / `\\owner{}` tags in "
                         "`slides/deck.tex` by `src/build_docs.py`._" if names
                         else "_Team not filled in: `slides/meta.yaml` → team._")
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


TEAM_MISSING_TEX = r"{\color{warn}team not filled in (slides/meta.yaml)}"


def tex_outputs(V: Values, B: dict) -> dict[Path, str]:
    """slides/gen/values.tex (\\V{key} for every key) and one file per slide table."""
    meta = B.get("meta", {})
    head = ["% generated by src/build_docs.py from results/: do not edit", ""]
    vals = [f"\\expandafter\\def\\csname v@{k}\\endcsname{{{tex(V[k])}}}" for k in sorted(V)]
    extra = [f"\\def\\basedir{{../{V['base.dir']}}}",
             # an empty team list must look incomplete on the title slide (and --strict fails on it)
             f"\\def\\teamnames{{{tex(', '.join(meta.get('team') or [])) or TEAM_MISSING_TEX}}}",
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


CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]+")
# Cyrillic words allowed in the English deliverables: quoted Russian data field names only (none so far)
CYRILLIC_ALLOW: set[str] = set()


def english_targets() -> list[Path]:
    """Deliverables that must be in English: docs, deck, generated slide files, deviations, upgrade summaries."""
    return ([ROOT / "README.md", ROOT / "slides" / "PLAN.md", DECK, ROOT / "DEVIATIONS.md"]
            + sorted(GEN.glob("*.tex")) + sorted((ROOT / "results" / "upgrades").glob("*/SUMMARY.md")))


def cyrillic_lines(text: str) -> list[str]:
    """Lines with Cyrillic words that are not on the allow-list."""
    out = []
    for i, ln in enumerate(text.splitlines(), 1):
        bad = [w for w in CYRILLIC_RE.findall(ln) if w not in CYRILLIC_ALLOW]
        if bad:
            out.append(f"line {i}: {' '.join(bad[:6])[:80]} | {ln.strip()[:100]}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--code-hash", action="store_true",
                    help="print the code-state hash (HEAD + diff of src/, tests/, config, Makefile, mk/) and exit")
    a = ap.parse_args(argv)
    if a.code_hash:
        print(code_state_hash())
        return 0
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
    V.put("dev.range", deviation_ranges(ROOT / "DEVIATIONS.md"))
    added = {f: subprocess.run(["git", "log", "--diff-filter=A", "--format=%h", "--", f], cwd=ROOT,
                               capture_output=True, text=True).stdout.split()
             for f in ["src/simulate.py", "results/simulations/sim_checks.csv"]}
    c_rules, c_res = (added[f][-1] if added[f] else None for f in added)
    V.put("git.sim_rules", f"The pass rules (`src/simulate.py`) were committed in the same commit as the first "
          f"simulation results (`{c_rules}`), so git does not show that they came first."
          if c_rules and c_rules == c_res else
          f"The pass rules were first committed in `{c_rules}`, the first simulation results in `{c_res}`.")
    mod, new = uncommitted()
    V.put("git.uncommitted_n", len(mod) + len(new))
    V.put("git.uncommitted_evidence", uncommitted_text(mod, new))
    first_deck = subprocess.run(["git", "log", "--diff-filter=A", "--format=%h %ad", "--date=format:%Y-%m-%d %H:%M",
                                 "--", "slides/deck.tex"], cwd=ROOT, capture_output=True, text=True).stdout.split("\n")
    first_deck = [x for x in first_deck if x.strip()]
    V.put("git.deck_first", f"{first_deck[-1].split(' ', 1)[1]} (commit `{first_deck[-1].split()[0]}`)"
          if first_deck else "not committed")
    repro_values(V, R)
    tests_log_values(V, R)
    mp = ROOT / "slides" / "meta.yaml"
    B["meta"] = (yaml.safe_load(V.text(mp)) or {}) if mp.exists() else {}
    blk = blocks(V, B, R)
    if a.list:
        for k in sorted(V):
            print(f"{k} = {V[k]}")
        print("blocks:", ", ".join(sorted(blk)))
        return 0

    rc, outputs, rendered = 0, {}, {}
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
        rendered[doc] = new
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
        rendered[path] = content
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
    # English-only deliverables (rendered content for the files this script writes, disk content otherwise)
    if a.strict:
        targets = list(dict.fromkeys(english_targets() + [q for q in rendered if q.parent == GEN]))
        for path in targets:
            if path not in rendered and not path.exists():
                continue
            bad = cyrillic_lines(rendered[path] if path in rendered else path.read_text())
            if bad:
                rc = 1
                print(f"{path.relative_to(ROOT)}: {len(bad)} line(s) with Cyrillic text:", file=sys.stderr)
                for h in bad[:20]:
                    print("   ", h, file=sys.stderr)
        if not (B["meta"].get("team") or []):
            print("slides/meta.yaml: team is empty (title slide and README show it as not filled in)",
                  file=sys.stderr)
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
