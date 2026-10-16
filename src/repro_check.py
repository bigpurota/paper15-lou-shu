"""Reproducibility evidence from two clean clones of this repository.

Mode 1, from-scratch clone (clone A: EMPTY data/, every download redone):
    python -I src/repro_check.py <clone_dir> [--log <clone_dir>/fresh_run.log]
  Writes results/repro/repro_check.csv (one row per compared file: max abs numeric difference for CSVs,
  byte identity for PNG figures), results/repro/repro_summary.json (clone commit, step timings and
  return codes, verdict) and copies the clone's logs (<clone_dir>/fresh_run.log, make_*.log) to
  results/repro/clone_logs/*.txt. The clone is expected to have run `make venv data tests p0` plus the
  upgrades (`upgrades-light`, `upgrade-hf_lambda`); upgrades it did not regenerate are listed as not_covered.
  The clone is compared with the outputs that results/LATEST points to when this script runs. Every
  `upgrade-hf_lambda` attempt in the clone (make_upgrade-hf_lambda*.log) is recorded with its failed downloads.

Mode 2, final clone (`make all` on a fresh clone; data/ a copy of the cached downloads or a link to them):
    python -I src/repro_check.py <clone_dir> --final --log <clone_dir>/final_run.log --make-log <clone_dir>/make_all.log
  Writes results/repro/final_clone/: summary.json (make all rc and seconds, the two pytest summaries of
  make all: code tests first, output-manifest contract last; every run/sim manifest with its git sha and
  dirty flag, canonical vs historical runs, whether data/ was a link), manifests.csv, base_vs_previous.csv
  (canonical base run against the previous base run with the same config, max abs diff per CSV),
  results_identity.csv (the clone's results/ files against this repository's, by sha256; a CSV that differs
  is also compared numerically) and the copied logs as .txt. The identity check is bookkeeping, not
  independent evidence: the canonical outputs were committed from that clone.

Absolute scratch-pad paths in copied logs are replaced by "<scratch>" and the home directory by "~".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "repro"
UPGRADES = ["amihud_noh", "edge_spread", "hf_lambda", "moex", "openap_us", "robustness", "sentiment"]
SKIP_COLS = ("sec", "time", "runtime")
# same rule as tests/test_manifests.py: an absolute path, or a relative one that walks into a home directory
ABS = re.compile(r"^(/|~|[A-Za-z]:[\\/])|/Users/|/home/")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _max_diff(a: Path, b: Path) -> tuple[str, float | None]:
    if not b.exists():
        return "missing_in_clone", None
    x, y = pd.read_csv(a), pd.read_csv(b)
    if a.name == "downloads.csv" and {"kind", "n", "gb"} <= set(x.columns) <= set(y.columns) | {"status"}:
        # per-run download log: rows split by status (cached vs downloaded in THIS run) depend on what
        # was already on disk; what must reproduce is the total number and size of files per kind
        x, y = (z.groupby("kind")[["n", "gb"]].sum().sort_index() for z in (x, y))
        if not x.index.equals(y.index):
            return "shape_or_columns_differ", None
        d = float((x - y).abs().max().max())
        return ("identical_totals" if d < 1e-9 else "numeric_diff"), d
    if ({"id", "value"} <= set(x.columns) and list(x.columns) == list(y.columns) and x["id"].is_unique
            and y["id"].is_unique and set(x["id"]) != set(y["id"])):
        # an id -> value table (report_diagnostics.csv) that gained or lost ids: the ids both sides share must agree
        common = sorted(set(x["id"]) & set(y["id"]))
        xv, yv = (z.set_index("id").loc[common, "value"] for z in (x, y))
        d = float((pd.to_numeric(xv, errors="coerce") - pd.to_numeric(yv, errors="coerce")).abs().max())
        return ("identical_common_rows" if d == 0 else "numeric_diff"), d
    if x.shape != y.shape or list(x.columns) != list(y.columns):
        return "shape_or_columns_differ", None
    num = [c for c in x.select_dtypes("number").columns if not any(s in c for s in SKIP_COLS)]
    other = [c for c in x.columns if c not in num and not any(s in c for s in SKIP_COLS)]
    text_same = not other or x[other].astype(str).equals(y[other].astype(str))
    d = float((x[num] - y[num]).abs().max().max()) if num else 0.0
    if d != 0:
        return "numeric_diff", d
    # text columns (wording, translation) may be edited after the clone's commit; numbers are what must reproduce
    return ("identical" if text_same else "numbers_identical_text_differs"), d


def _csv_row(group: str, rel: str, a: Path, b: Path) -> dict:
    """One compared CSV. _max_diff leaves out the columns whose names contain a SKIP_COLS substring; those columns
    are listed here and checked for exact equality on their own, so the report can say what was left out."""
    st, d = _max_diff(a, b)
    row = {"group": group, "kind": "csv", "file": rel, "status": st, "max_abs_diff": d,
           "skipped_cols": "", "skipped_identical": None}
    try:
        x = pd.read_csv(a)
        y = pd.read_csv(b) if b.exists() else None
    except Exception:
        return row
    sk = [c for c in x.columns if any(s in c for s in SKIP_COLS)]
    if sk:
        row["skipped_cols"] = ";".join(sk)
        row["skipped_identical"] = bool(y is not None and all(c in y.columns for c in sk) and len(x) == len(y)
                                        and x[sk].astype(str).equals(y[sk].astype(str)))
    return row


def _png_row(group: str, rel: str, a: Path, b: Path) -> dict:
    st = "missing_in_clone" if not b.exists() else ("identical" if _sha(a) == _sha(b) else "bytes_differ")
    return {"group": group, "kind": "png", "file": rel, "status": st, "max_abs_diff": None}


def _redact(text: str, clone: Path) -> str:
    scratch = str(clone.parent)
    for s in {scratch, str(Path(scratch).resolve())}:
        text = text.replace(s, "<scratch>")
    text = re.sub(r"/private/tmp/claude-\d+/[^/\s]+/[^/\s]+/scratchpad", "<scratch>", text)
    return text.replace(str(Path.home()), "~")


def _copy_logs(clone: Path, logs: list[Path], dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for p in logs:
        if p.exists():
            q = dest / (p.stem + ".txt")
            q.write_text(_redact(p.read_text(errors="replace"), clone))
            out.append(str(q.relative_to(ROOT)))
    return out


def _steps(log: Path | None) -> list[dict]:
    if not log or not log.exists():
        return []
    return [{"target": m.group(1), "rc": int(m.group(2)), "seconds": int(m.group(3))}
            for m in re.finditer(r"=== make (\S+) rc=(\d+) took (\d+)s", log.read_text())]


def _hf_attempts(clone: Path) -> list[dict]:
    """Every `make upgrade-hf_lambda` attempt in the clone (oldest first): seconds and failed downloads."""
    out = []
    for p in sorted(clone.glob("make_upgrade-hf_lambda*.log"), key=lambda q: q.stat().st_mtime):
        txt = p.read_text(errors="replace")
        failed = {k: int(n) for k, n in re.findall(r"\('(\w+)', 'failed_after_retries'\): (\d+)", txt)}
        sec = re.findall(r"written to \S+ \((\d+)s\)", txt)
        out.append({"log": p.stem, "seconds": int(sec[-1]) if sec else None, "failed_downloads": failed,
                    "n_failed": int(sum(failed.values()))})
    return out


def _git(clone: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(clone), *args], capture_output=True, text=True).stdout.strip()


def _walk(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from _walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from _walk(v)
    else:
        yield x


# ------------------------------------------------------------------------------------ mode 1: clone A
def fresh(clone: Path, log: Path | None) -> dict:
    rows = []
    da = pd.read_parquet(ROOT / "data/cache/daily.parquet")
    db = pd.read_parquet(clone / "data/cache/daily.parquet")
    cols = [c for c in da.columns if c in db.columns]
    da = da[cols].sort_values(["symbol", "date"]).reset_index(drop=True)
    db = db[cols].sort_values(["symbol", "date"]).reset_index(drop=True)
    rows.append({"group": "data", "kind": "parquet", "file": "data/cache/daily.parquet",
                 "status": "identical" if da.equals(db) else "differs", "max_abs_diff": None,
                 "rows_repo": len(da), "rows_clone": len(db)})

    ra = (ROOT / "results/LATEST").read_text().strip()
    rb = (clone / "results/LATEST").read_text().strip()
    for f in sorted((ROOT / "results" / ra).glob("*.csv")):
        rows.append(_csv_row("base", f"results/<run>/{f.name}", f, clone / "results" / rb / f.name))
    for f in sorted((ROOT / "results" / ra).glob("*.png")):
        rows.append(_png_row("base", f"results/<run>/{f.name}", f, clone / "results" / rb / f.name))
    for f in sorted((ROOT / "results/simulations").glob("*.csv")):
        rows.append(_csv_row("simulations", f"results/simulations/{f.name}", f, clone / "results/simulations" / f.name))
    not_covered = []
    for u in UPGRADES:
        # results/ is tracked in git, so the clone starts with OUR committed outputs: an upgrade counts as
        # reproduced only if the clone regenerated it (its run_manifest.json differs from the cloned commit)
        man = f"results/upgrades/{u}/run_manifest.json"
        regenerated = subprocess.run(["git", "-C", str(clone), "diff", "--quiet", "HEAD", "--", man]).returncode == 1
        if not (clone / man).exists() or not regenerated:
            not_covered.append(u)
            continue
        for f in sorted((ROOT / "results/upgrades" / u).glob("*.csv")):
            rows.append(_csv_row(u, f"results/upgrades/{u}/{f.name}", f, clone / "results/upgrades" / u / f.name))
    df = pd.DataFrame(rows)

    steps = _steps(log)
    sha = _git(clone, "rev-parse", "HEAD")
    csv = df[df.kind == "csv"]
    numeric_ok = csv["status"].isin(["identical", "numbers_identical_text_differs", "identical_totals",
                                     "identical_common_rows"]) | (
        (csv["status"] == "numeric_diff") & (csv["max_abs_diff"] < 1e-9))
    png = df[df.kind == "png"]
    data_ok = bool((df[df.kind == "parquet"].status == "identical").all())
    rc_ok = not steps or all(s["rc"] == 0 for s in steps)
    man_b = json.loads((ROOT / "results" / ra / "run_manifest.json").read_text())
    summary = {
        "mode": "from_scratch_clone",
        "clone_dir": "<scratch>/" + clone.name,
        "clone_commit": sha,
        "clone_data_dir": "empty at start; every download redone by the clone",
        "code_note": (f"the clone ran the code of commit {sha[:7]}; it is compared with this repository's canonical "
                      f"outputs, which were produced by commit {man_b['git']['sha'][:7]} (base run {ra})"),
        "repo_base_run": ra, "clone_base_run": rb,
        "steps": steps, "all_steps_rc0": bool(steps) and rc_ok,
        "n_files_compared": int(len(df)), "n_csv_compared": int(len(csv)),
        "n_identical": int((csv["status"] == "identical").sum()),
        "n_numbers_identical_text_differs": int((csv["status"] == "numbers_identical_text_differs").sum()),
        "n_identical_totals": int((csv["status"] == "identical_totals").sum()),
        "identical_totals_note": ("a download log (hf_lambda/downloads.csv): totals per kind agree; its rows are split "
                                  "by cache status, which depends on what was already on disk"),
        "n_identical_common_rows": int((csv["status"] == "identical_common_rows").sum()),
        "identical_common_rows_note": ("an id -> value table with ids added on one side (report_diagnostics.csv after "
                                       "a report.py change): the values of every id both sides share are identical"),
        "n_not_identical": int((~numeric_ok).sum()),
        "skipped_cols_rule": ("every CSV comparison leaves out the columns whose names contain one of "
                              + ", ".join(f"'{s}'" for s in SKIP_COLS) + " (meant for run-time columns); "
                              "'identical' means identical apart from them; they are checked separately here"),
        "skipped_cols_differ": [{"file": r.file, "cols": r.skipped_cols} for r in csv.itertuples()
                                if r.skipped_cols and r.skipped_identical is False],
        "skipped_cols_identical": [{"file": r.file, "cols": r.skipped_cols} for r in csv.itertuples()
                                   if r.skipped_cols and r.skipped_identical is True],
        "n_png_compared": int(len(png)), "n_png_identical": int((png.status == "identical").sum()),
        "daily_identical": data_ok,
        "hf_attempts": (hf := _hf_attempts(clone)),
        "hf_note": (f"{len(hf)} `make upgrade-hf_lambda` attempts; every earlier attempt with failed downloads still "
                    f"exited 0 (the hf_lambda code at {sha[:7]} did not fail on failed downloads); only the last "
                    f"attempt, with {hf[-1]['n_failed']} failed downloads, wrote the outputs that are compared"
                    if len(hf) > 1 else "one `make upgrade-hf_lambda` attempt"),
        "upgrades_compared": [u for u in UPGRADES if u not in not_covered],
        "not_covered": not_covered,
        "not_compared": ["PDF figures (they embed a creation date)", "README/PLAN/deck (rendered from the CSVs)",
                         "the test ledger (written by `make ledger`, not run in this clone)"],
        "daily_sha256_repo": _sha(ROOT / "data/cache/daily.parquet")[:16],
        "verdict": ("differences" if not (numeric_ok.all() and rc_ok and data_ok)
                    else "reproduced" if not not_covered else "reproduced_except_not_covered"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "repro_check.csv", index=False)
    logs = ([log] if log else []) + sorted(clone.glob("make_*.log"))
    summary["logs"] = _copy_logs(clone, logs, OUT / "clone_logs")
    (OUT / "repro_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# ------------------------------------------------------------------------------------ mode 2: clone B
def final(clone: Path, log: Path | None, make_log: Path | None) -> dict:
    out = OUT / "final_clone"
    out.mkdir(parents=True, exist_ok=True)
    steps = _steps(log)
    mk_all = [s for s in steps if s["target"] == "all"]
    mtxt = make_log.read_text(errors="replace") if make_log and make_log.exists() else ""
    # make all runs pytest twice: the code tests first (`make tests`, without tests/test_manifests.py) and
    # the output-manifest contract last (`check-outputs`, tests/test_manifests.py on the outputs just written)
    tests = re.findall(r"^(\d+ passed[^\n]*?) in [0-9.]+s", mtxt, re.M)

    canonical = (clone / "results/LATEST").read_text().strip()
    repo_latest = (ROOT / "results/LATEST").read_text().strip()
    rows = []
    paths = sorted((ROOT / "results").glob("**/run_manifest.json")) + sorted((ROOT / "results").glob("**/sim_manifest.json"))
    for p in paths:
        rel = p.relative_to(ROOT / "results")
        if rel.parts[0] == "repro":
            continue
        m = json.loads(p.read_text())
        g = m.get("git") or {}
        is_base = rel.parts[0] not in ("upgrades", "simulations")
        role = "canonical" if (not is_base or rel.parts[0] == canonical) else "historical"
        leaks = sorted({v for v in _walk(m) if isinstance(v, str) and ABS.search(v)})
        rows.append({"manifest": f"results/{rel}", "run_id": m.get("run_id"), "role": role,
                     "base_run_id": m.get("base_run_id"), "git_sha": (g.get("sha") or m.get("git_sha") or "")[:7],
                     "dirty": g.get("dirty", m.get("git_dirty")), "dirty_scope": g.get("dirty_scope", "not recorded"),
                     "n_home_paths": len(leaks)})
    mf = pd.DataFrame(rows)
    mf.to_csv(out / "manifests.csv", index=False)
    can = mf[mf.role == "canonical"]
    run_sha = sorted(set(can.git_sha))

    # canonical base run against the previous base run with the same config hash (its last 8 hex in the id)
    tag = canonical.split("_")[-1]
    prev = sorted(p.name for p in (ROOT / "results").iterdir()
                  if p.is_dir() and p.name.endswith("_" + tag) and p.name < canonical)
    cmp_rows = []
    if prev:
        for f in sorted((ROOT / "results" / canonical).glob("*.csv")):
            st, d = _max_diff(f, ROOT / "results" / prev[-1] / f.name)
            cmp_rows.append({"file": f.name, "status": st.replace("_in_clone", "_in_previous"), "max_abs_diff": d})
    cmp = pd.DataFrame(cmp_rows, columns=["file", "status", "max_abs_diff"])
    cmp.to_csv(out / "base_vs_previous.csv", index=False)

    # the clone's results/ against this repository's (the canonical outputs were committed from the clone)
    # (results/repro/ is written by this script and results/docs_manifest.json by every `make docs`: both excluded)
    ident = []
    for p in sorted((clone / "results").rglob("*")):
        rel = p.relative_to(clone / "results")
        if p.is_file() and rel.parts[0] != "repro" and str(rel) != "docs_manifest.json":
            q = ROOT / "results" / p.relative_to(clone / "results")
            st, d = ("missing_in_repo", None) if not q.exists() else (("identical", 0.0) if _sha(p) == _sha(q)
                                                                       else ("differs", None))
            if st == "differs" and p.suffix == ".csv":
                st, d = _max_diff(q, p)
            ident.append({"file": str(q.relative_to(ROOT)), "status": st, "max_abs_diff": d})
    idf = pd.DataFrame(ident)
    idf.to_csv(out / "results_identity.csv", index=False)

    data_link = (clone / "data").is_symlink()
    summary = {
        "mode": "final_clone_make_all",
        "clone_dir": "<scratch>/" + clone.name,
        "clone_head_now": _git(clone, "rev-parse", "--short", "HEAD"),
        "clone_head_note": "the canonical outputs were committed from this clone after the run, so its HEAD is "
                           "the commit that holds them; the code that ran is the git sha in the manifests",
        "code_commit_run": run_sha[0] if len(run_sha) == 1 else run_sha,
        "data_dir_was_link": data_link,
        "data_note": ("data/ was a symbolic link to the main repository's cached downloads (data/raw, data/cache): "
                      "this run proves that the committed code regenerates every result from cached raw data; "
                      "that the downloads themselves reproduce is shown by the from-scratch clone "
                      "(results/repro/repro_summary.json)") if data_link else "data/ was a directory in the clone",
        "make_all": mk_all[0] if mk_all else None,
        "tests_summary": tests[0] if tests else None,
        "tests_contract_summary": tests[-1] if len(tests) > 1 else None,
        "tests_note": ("make all runs the code tests first (pytest --ignore=tests/test_manifests.py; tests_summary) and "
                       "the output-manifest contract last (check-outputs: pytest tests/test_manifests.py on the outputs "
                       "it has just written; tests_contract_summary); the contract's xfails are historical base runs "
                       "whose manifests hold absolute or home-directory paths"),
        "canonical_base_run": canonical, "repo_latest": repo_latest,
        "n_manifests": int(len(mf)), "n_canonical": int(len(can)),
        "canonical_all_same_commit_clean": bool(len(run_sha) == 1 and (can.dirty == False).all()),  # noqa: E712
        "historical_runs": mf[mf.role == "historical"][["run_id", "git_sha", "dirty", "n_home_paths"]].to_dict("records"),
        "n_canonical_with_home_paths": int((can.n_home_paths > 0).sum()),
        "home_path_note": ("relative paths in some canonical manifests resolve into a home directory (../../Users/...), "
                           "which tests/test_manifests.py rejects as absolute paths"
                           if (can.n_home_paths > 0).any() else
                           "no canonical manifest holds an absolute or home-directory path; historical base runs "
                           "with such paths: " + (", ".join(mf[(mf.role == "historical") & (mf.n_home_paths > 0)].run_id)
                                                  or "none")),
        "base_compared_with": prev[-1] if prev else None,
        "base_n_csv": int(len(cmp)),
        "base_max_abs_diff": float(cmp.max_abs_diff.max()) if len(cmp) and cmp.max_abs_diff.notna().all() else None,
        "base_n_not_identical": int((~cmp.status.isin(["identical", "numbers_identical_text_differs", "identical_totals",
                                                         "identical_common_rows"])).sum()),
        "base_n_text_or_rows": int(cmp.status.isin(["numbers_identical_text_differs", "identical_common_rows"]).sum()),
        "results_files_in_clone": int(len(idf)),
        "results_files_identical_to_repo": int((idf.status == "identical").sum()) if len(idf) else 0,
        "results_files_differing": idf[idf.status != "identical"][["file", "status"]].to_dict("records") if len(idf) else [],
        "identity_note": ("bookkeeping, not independent evidence: the canonical outputs were committed from this clone; "
                          "a file that differs was rewritten in this repository after the clone run"),
    }
    summary["logs"] = _copy_logs(clone, [x for x in (log, make_log) if x], out)
    # the verdict needs make all rc 0, one clean commit in every canonical manifest, an identical base run against
    # the previous one, and no numeric difference or missing file between the clone's results/ and ours (text-only
    # rewrites made here after the clone run, e.g. report wording, are listed in results_files_differing)
    num_bad = idf.status.isin(["numeric_diff", "shape_or_columns_differ", "missing_in_repo"]).sum() if len(idf) else 0
    ok = (summary["make_all"] and summary["make_all"]["rc"] == 0 and summary["canonical_all_same_commit_clean"]
          and summary["base_n_not_identical"] == 0 and num_bad == 0)
    summary["verdict"] = "reproduced" if ok else "differences"
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clone")
    ap.add_argument("--log")
    ap.add_argument("--final", action="store_true", help="clone B: `make all` on a fresh clone")
    ap.add_argument("--make-log")
    a = ap.parse_args()
    clone = Path(a.clone).absolute()
    log = Path(a.log) if a.log else None
    if a.final:
        s = final(clone, log, Path(a.make_log) if a.make_log else None)
    else:
        s = fresh(clone, log)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
