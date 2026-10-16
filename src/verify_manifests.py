"""Recompute the content hashes recorded in the canonical run manifests and compare them with the files here.

Needs no git, so it works on the unpacked ZIP. Canonical manifests: the base run that results/LATEST points to
(run_manifest.json and report_manifest.json), results/simulations/sim_manifest.json and
results/upgrades/*/run_manifest.json. One more row checks the pre-registration: the config_sha256 of the first
real-data run (the earliest base run manifest) against docs/preregistration/config_preregistered.yaml.

Checks per manifest:
    config  sha256 of the config file the manifest names (config.yaml by default) against config_sha256
    code    base run: sha256 over (file name, bytes) of src/*.py in sorted order (src/run.py code_hash);
            simulations: sha256 of the concatenated bytes of src/*.py (src/simulate.py);
            the others: the sha256 of every file listed in code_sha256
    data    every input with a recorded path and sha256: rechecked when the file is present, otherwise reported
            as absent (data/ is not in the ZIP); files under results/ are in the ZIP and are always rechecked

Status per row: match, MISMATCH, absent (file not here), not rechecked (no path recorded, or an aggregate hash).

Usage:
    python -I src/verify_manifests.py [--results-dir results] [--csv out.csv]
Exit code 1 if any recomputed hash differs from the manifest (MISMATCH), 0 otherwise.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "docs" / "preregistration" / "config_preregistered.yaml"
FIELDS = ["manifest", "check", "item", "status", "recorded", "recomputed"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hash_named(src: Path) -> str:
    """src/run.py code_hash: sha256 over (file name, bytes) of src/*.py in sorted order."""
    h = hashlib.sha256()
    for p in sorted(src.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def code_hash_concat(src: Path) -> str:
    """src/simulate.py: sha256 of the concatenated bytes of src/*.py in sorted order."""
    return hashlib.sha256(b"".join(p.read_bytes() for p in sorted(src.glob("*.py")))).hexdigest()


def rel(p: Path) -> str:
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def compare(rows: list, man: str, check: str, item: str, recorded, path: Path | None, recompute=None) -> None:
    """Append one row. `recompute` overrides hashing `path` (aggregate code hashes)."""
    if not isinstance(recorded, str) or len(recorded) != 64:
        rows.append(dict(manifest=man, check=check, item=item, status="not rechecked", recorded=str(recorded)[:16],
                         recomputed=""))
        return
    if recompute is None and (path is None or not path.is_file()):
        rows.append(dict(manifest=man, check=check, item=item, status="absent" if path is not None else "not rechecked",
                         recorded=recorded[:16], recomputed=""))
        return
    got = recompute() if recompute is not None else sha256(path)
    rows.append(dict(manifest=man, check=check, item=item, status="match" if got == recorded else "MISMATCH",
                     recorded=recorded[:16], recomputed=got[:16]))


def code_file(key: str) -> Path:
    """Keys of a per-file code_sha256: 'src/x.py' (repo-relative) or 'x.py' (relative to src/)."""
    return ROOT / key if "/" in key else ROOT / "src" / key


def data_pairs(m: dict, mdir: Path) -> list[tuple[str, str, Path | None]]:
    """(item, recorded sha256, path or None) for every data/input hash in a manifest."""
    out: list[tuple[str, str, Path | None]] = []
    if isinstance(m.get("data_source"), str) and isinstance(m.get("data_sha256"), str):
        out.append((m["data_source"], m["data_sha256"], ROOT / m["data_source"]))
    for key in ["inputs", "input_data"]:                          # {name: {path|file, sha256}}
        for name, v in (m.get(key) or {}).items():
            if isinstance(v, dict) and "sha256" in v:
                p = v.get("path") or v.get("file")
                out.append((p or name, v["sha256"], ROOT / p if p else None))
    for p, s in (m.get("input_sha256") or {}).items():            # {repo-relative path: sha256}
        out.append((p, s, ROOT / p))
    for name, s in (m.get("inputs_sha256") or {}).items():        # report stage: files of the base run folder
        out.append((rel(mdir / name), s, mdir / name))
    for name, s in (m.get("simulations_sha256") or {}).items():
        out.append((f"results/simulations/{name}", s, ROOT / "results" / "simulations" / name))
    if isinstance(m.get("panel_cache"), str) and isinstance(m.get("panel_cache_sha256"), str):
        out.append((m["panel_cache"], m["panel_cache_sha256"], ROOT / m["panel_cache"]))
    d = m.get("data")                                              # {x: path, x_sha256: sha} (sentiment)
    if isinstance(d, dict):
        for k, s in d.items():
            if k.endswith("_sha256"):
                stem = k[: -len("_sha256")]
                p = d.get(stem) or d.get(stem + "_json") or d.get(stem + "_cache")
                out.append((p or stem, s, ROOT / p if isinstance(p, str) else None))
    ds = m.get("data_sha256")
    if isinstance(ds, dict):                                       # {name: sha} with paths in data_paths or elsewhere
        paths = m.get("data_paths") or {}
        for name, s in ds.items():
            if not isinstance(s, str):
                continue
            p = paths.get(name) or (m.get(name) if isinstance(m.get(name), str) else None)
            if p is None and name.endswith(".parquet"):           # MOEX: file names in the raw cache folder
                p = f"data/cache/{name}"
            if p is None and name == "edge_monthly":              # src/hf_lambda.py: CACHE / edge_monthly.parquet
                p = "data/cache/edge_monthly.parquet"
            if p is None and name == "base_factors_csv" and m.get("base_run_id"):   # src/spread_edge.py
                p = f"results/{m['base_run_id']}/factors.csv"
            out.append((p or name, s, ROOT / p if p else None))
    return out


def verify(results_dir: Path | None = None) -> list[dict]:
    R = Path(results_dir or ROOT / "results")
    latest = R / (R / "LATEST").read_text().strip()
    mans = [latest / "run_manifest.json", latest / "report_manifest.json", R / "simulations" / "sim_manifest.json"]
    mans += sorted(R.glob("upgrades/*/run_manifest.json"))
    rows: list[dict] = []
    for mp in mans:
        if not mp.exists():
            rows.append(dict(manifest=rel(mp), check="manifest", item="", status="absent", recorded="", recomputed=""))
            continue
        m = json.loads(mp.read_text())
        name = rel(mp)
        cfg = ROOT / (m.get("config_path") or "config.yaml")
        compare(rows, name, "config", rel(cfg), m.get("config_sha256"), cfg)
        code = m.get("code_sha256")
        if isinstance(code, str):
            compare(rows, name, "code", "src/*.py (names and bytes)", code, None, lambda: code_hash_named(ROOT / "src"))
        elif isinstance(code, dict) and len(code) == 1 and next(iter(code)).startswith("src/*.py"):
            compare(rows, name, "code", "src/*.py (concatenated)", next(iter(code.values())), None,
                    lambda: code_hash_concat(ROOT / "src"))
        elif isinstance(code, dict):
            for k, s in code.items():
                compare(rows, name, "code", rel(code_file(k)), s, code_file(k))
        for item, s, p in data_pairs(m, mp.parent):
            compare(rows, name, "data", item, s, p)
    # pre-registration: the first real-data run used the pre-registered config
    first = []
    for mp in R.glob("*/run_manifest.json"):
        m = json.loads(mp.read_text())
        if "created_utc" in m:
            first.append((m["created_utc"], mp, m))
    if first:
        _, mp, m = min(first, key=lambda x: x[0])
        compare(rows, rel(mp), "pre-registration", rel(PREREG), m.get("config_sha256"), PREREG)
    return rows


def summary(rows: list[dict]) -> dict:
    """Counts by check and status, plus the manifests whose code hashes do not match."""
    out: dict = {}
    for r in rows:
        out.setdefault(r["check"], {}).setdefault(r["status"], 0)
        out[r["check"]][r["status"]] += 1
    out["code_mismatch_manifests"] = sorted({r["manifest"] for r in rows
                                             if r["check"] == "code" and r["status"] == "MISMATCH"})
    out["n_manifests"] = len({r["manifest"] for r in rows if r["check"] != "pre-registration"})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--csv", help="also write the table to this CSV file")
    a = ap.parse_args(argv)
    rows = verify(Path(a.results_dir))
    w = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in FIELDS}
    print("  ".join(k.ljust(w[k]) for k in FIELDS))
    for r in rows:
        print("  ".join(str(r[k]).ljust(w[k]) for k in FIELDS))
    s = summary(rows)
    print()
    for check in ["config", "code", "data", "pre-registration"]:
        print(f"{check}: " + ", ".join(f"{n} {st}" for st, n in sorted(s.get(check, {}).items())))
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=FIELDS)
            wr.writeheader()
            wr.writerows(rows)
    bad = [r for r in rows if r["status"] == "MISMATCH"]
    if bad:
        print(f"\n{len(bad)} hash(es) differ from the manifests (code or inputs changed after the run)")
        return 1
    print("\nevery recomputed hash matches its manifest (absent: inputs not present here, e.g. data/ in the ZIP)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
