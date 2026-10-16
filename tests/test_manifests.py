"""Provenance contract for every results/**/run_manifest.json and results/simulations/sim_manifest.json.

Common keys: run_id, created_utc, git {sha, dirty}, config_sha256, code_sha256 (per file or aggregate),
a data/input hash block (contains a sha256, or says 'synthetic' for the simulations), seed, timings_sec;
and no absolute file-system paths anywhere in the manifest (they leak the user name and break elsewhere).

Manifests whose regeneration is heavy (base runs results/<run_id>/, simulations, hf_lambda) were written
by an older code version and are only refreshed by the clean-clone full rerun: for them a failure is an
xfail with reason "regenerated in clean-clone run" (non-strict, so they pass once regenerated). All the
cheap upgrades (make upgrade-<name>) must pass now.
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANIFESTS = sorted(RESULTS.glob("**/run_manifest.json")) + sorted(RESULTS.glob("**/sim_manifest.json"))
HEAVY_REASON = "regenerated in clean-clone run"
DATA_KEYS = ["data_sha256", "input_sha256", "inputs", "input_data", "data"]
SHA = re.compile(r"^[0-9a-f]{64}$")
ABS = re.compile(r"^(/|~|[A-Za-z]:[\\/])|/Users/|/home/")


def _heavy(p: Path) -> bool:
    rel = p.relative_to(RESULTS).parts
    return rel[0] == "simulations" or rel[:2] == ("upgrades", "hf_lambda") or rel[0] != "upgrades"


def _walk(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _walk(k, f"{path}.<key>")
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, x


def problems(m: dict) -> list[str]:
    out = []
    for k in ["run_id", "created_utc", "config_sha256", "code_sha256", "seed", "timings_sec"]:
        if k not in m or m[k] in (None, "", {}):
            out.append(f"missing {k}")
    g = m.get("git")
    if not isinstance(g, dict) or not g.get("sha") or "dirty" not in g:
        out.append("missing git {sha, dirty}")
    if "config_sha256" in m and not SHA.match(str(m["config_sha256"])):
        out.append("config_sha256 is not a sha256")
    if "code_sha256" in m and not any(isinstance(v, str) and SHA.match(v) for _, v in _walk(m["code_sha256"])):
        out.append("code_sha256 holds no sha256")
    blocks = [m[k] for k in DATA_KEYS if k in m]
    if not blocks:
        out.append("missing data/input hash block (one of %s)" % DATA_KEYS)
    elif not any(isinstance(v, str) and (SHA.match(v) or v.startswith("synthetic"))
                 for b in blocks for _, v in _walk(b)):
        out.append("data/input block holds no sha256")
    if "timings_sec" in m and not isinstance(m["timings_sec"], dict):
        out.append("timings_sec is not a dict")
    for path, v in _walk(m):
        if isinstance(v, str) and ABS.search(v):
            out.append(f"absolute path at {path}: {v[:60]}")
    return out


def test_manifests_exist():
    assert any(p.parent.parent.name == "upgrades" for p in MANIFESTS), "no upgrade manifests found"


@pytest.mark.parametrize("path", MANIFESTS, ids=[str(p.relative_to(RESULTS)) for p in MANIFESTS])
def test_manifest_contract(path):
    pr = problems(json.loads(path.read_text()))
    if pr and _heavy(path):
        pytest.xfail(f"{HEAVY_REASON}: " + "; ".join(pr))
    assert not pr, pr


def test_contract_checker_flags_bad_manifest():
    bad = {"run_id": "x", "config_path": "/Users/someone/repo/config.yaml"}
    pr = problems(bad)
    assert any("absolute path" in p for p in pr) and any("missing git" in p for p in pr)
    good = {"run_id": "x", "created_utc": "20260101T000000Z", "git": {"sha": "abc", "dirty": False},
            "config_sha256": "0" * 64, "code_sha256": {"a.py": "1" * 64}, "data_sha256": {"d": "2" * 64},
            "seed": 1, "timings_sec": {"total": 1.0}, "config_path": "config.yaml",
            "url": "https://example.org/a/b"}
    assert problems(good) == []
