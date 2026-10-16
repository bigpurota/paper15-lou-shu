"""Compare a from-scratch clean-clone run with this repository's results (reproducibility evidence).

Usage:  python -I src/repro_check.py <clone_dir> [--log <clone_dir>/fresh_run.log]
Writes: results/repro/repro_check.csv  (one row per compared file: max abs numeric difference)
        results/repro/repro_summary.json (clone commit, step timings and return codes, verdict)
The clone must have been built with `make venv data tests p0` plus the upgrades (`upgrades-light`,
`upgrade-hf_lambda`) in an EMPTY data/ directory; upgrades the clone did not run are listed as not_covered.
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


def _max_diff(a: Path, b: Path) -> tuple[str, float | None]:
    if not b.exists():
        return "missing_in_clone", None
    x, y = pd.read_csv(a), pd.read_csv(b)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clone")
    ap.add_argument("--log")
    a = ap.parse_args()
    clone = Path(a.clone).resolve()
    rows = []

    da = pd.read_parquet(ROOT / "data/cache/daily.parquet")
    db = pd.read_parquet(clone / "data/cache/daily.parquet")
    cols = [c for c in da.columns if c in db.columns]
    da = da[cols].sort_values(["symbol", "date"]).reset_index(drop=True)
    db = db[cols].sort_values(["symbol", "date"]).reset_index(drop=True)
    rows.append({"group": "data", "file": "data/cache/daily.parquet",
                 "status": "identical" if da.equals(db) else "differs", "max_abs_diff": None,
                 "rows_repo": len(da), "rows_clone": len(db)})

    ra = (ROOT / "results/LATEST").read_text().strip()
    rb = (clone / "results/LATEST").read_text().strip()
    for f in sorted((ROOT / "results" / ra).glob("*.csv")):
        st, d = _max_diff(f, clone / "results" / rb / f.name)
        rows.append({"group": "base", "file": f"results/<run>/{f.name}", "status": st, "max_abs_diff": d})
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
            st, d = _max_diff(f, clone / "results/upgrades" / u / f.name)
            rows.append({"group": u, "file": f"results/upgrades/{u}/{f.name}", "status": st, "max_abs_diff": d})
    df = pd.DataFrame(rows)

    steps = []
    if a.log and Path(a.log).exists():
        for m in re.finditer(r"=== make (\S+) rc=(\d+) took (\d+)s", Path(a.log).read_text()):
            steps.append({"target": m.group(1), "rc": int(m.group(2)), "seconds": int(m.group(3))})
    sha = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    numeric_ok = df["status"].isin(["identical", "numbers_identical_text_differs"]) | (
        (df["status"] == "numeric_diff") & (df["max_abs_diff"] < 1e-9))
    summary = {
        "clone_commit": sha, "repo_base_run": ra, "clone_base_run": rb,
        "steps": steps, "all_steps_rc0": bool(steps) and all(s["rc"] == 0 for s in steps),
        "n_files_compared": int(len(df)), "n_identical": int((df["status"] == "identical").sum()),
        "n_numbers_identical_text_differs": int((df["status"] == "numbers_identical_text_differs").sum()),
        "n_not_identical": int((~numeric_ok).sum()),
        "upgrades_compared": [u for u in UPGRADES if u not in not_covered],
        "not_covered": not_covered,
        "daily_sha256_repo": hashlib.sha256((ROOT / "data/cache/daily.parquet").read_bytes()).hexdigest()[:16],
        "verdict": ("differences" if not (numeric_ok.all() and (not steps or all(s["rc"] == 0 for s in steps)))
                    else "reproduced" if not not_covered else "reproduced_except_not_covered"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "repro_check.csv", index=False)
    (OUT / "repro_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
