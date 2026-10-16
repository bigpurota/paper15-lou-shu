"""End-to-end smoke test: run.py -> report.py -> figures.py on a small synthetic daily panel."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from run import synthetic_daily

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _py(script: str, *args: str, cwd: Path) -> subprocess.CompletedProcess:
    r = subprocess.run([sys.executable, "-I", str(SRC / script), *args], cwd=cwd,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout}\n{r.stderr}"
    return r


def test_pipeline_end_to_end(tmp_path):
    daily = synthetic_daily(n_symbols=60, start="2018-01-01", end="2020-06-30", seed=1)
    daily_path = tmp_path / "daily.parquet"
    daily.to_parquet(daily_path, index=False)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["period"] = {"return_start": "2019-01", "return_end": "2020-06"}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    res, cache = tmp_path / "results", tmp_path / "cache"

    _py("run.py", str(cfg_path), "--daily", str(daily_path), "--results-dir", str(res),
        "--cache-dir", str(cache), "--tag", "smoke", "--no-placebo", cwd=tmp_path)
    run_id = (res / "LATEST").read_text().strip()
    rd = res / run_id
    assert run_id.endswith("_smoke")

    man = json.loads((rd / "run_manifest.json").read_text())
    for k in ["run_id", "git", "config_sha256", "code_sha256", "data_sha256", "seed", "versions",
              "timings_sec", "rows", "peak_rss_mb", "tables"]:
        assert k in man, k
    assert man["run_id"] == run_id and man["seed"] == cfg["seed"]
    assert len(man["data_sha256"]) == 64 and len(man["config_sha256"]) == 64
    assert {"load", "build_panel", "run_tests", "write"} <= set(man["timings_sec"])
    assert man["rows"]["daily_rows"] == len(daily) and man["rows"]["panel_months"] == 18
    for t in man["tables"]:
        assert (rd / t).exists(), t
    for t in ["T1_corr.csv", "T3_sorts.csv", "T4_fm.csv", "T3_portfolios.csv", "panel_summary.csv"]:
        assert (rd / t).exists(), t
    assert list(cache.glob("panel_*.parquet")), "panel parquet should go to the cache dir"
    assert not list(rd.glob("*.parquet")), "full panel must not be stored in results/"

    _py("report.py", "--results-dir", str(res), cwd=tmp_path)
    pv = pd.read_csv(rd / "paper_vs_us.csv")
    assert len(pv) >= 15 and pv["us"].notna().all()
    assert set(pv["match"]) <= {"yes", "sign only", "no", "n/a"}
    assert (rd / "paper_vs_us.md").read_text().count("\n|") >= len(pv) + 1
    tex = (rd / "tex" / "numbers.tex").read_text()
    names = re.findall(r"\\newcommand\{\\([^}]*)\}", tex)
    assert names and all(n.isalpha() for n in names) and len(names) == len(set(names))

    _py("figures.py", "--results-dir", str(res), cwd=tmp_path)
    for f in ["fig_quintiles", "fig_quintiles_demeaned", "fig_hl_cumulative"]:
        for ext in ["png", "pdf"]:
            assert (rd / f"{f}.{ext}").stat().st_size > 1000, f


def test_match_rule():
    from report import match
    assert match("t", (0.56, 2.36), (0.9, 2.5)) == "yes"
    assert match("t", (0.56, 2.36), (0.9, 1.0)) == "sign only"
    assert match("t", (0.56, 2.36), (-0.9, -3.0)) == "no"
    assert match("t", (-0.17, -1.05), (0.3, 0.5)) == "yes"          # both insignificant
    assert match("t", (-0.17, -1.05), (-0.5, -2.5)) == "sign only"
    assert match("level", (0.899, None), (0.85, None)) == "yes"
    assert match("level", (116.6, None), (20.0, None)) == "sign only"
    assert match("level", (0.394, None), (-0.1, None)) == "no"
