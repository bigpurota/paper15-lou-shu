"""src/verify_manifests.py (`make verify`): recomputed hashes match untouched files and flag edited ones."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import verify_manifests as vm  # noqa: E402


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _repo(tmp: Path) -> Path:
    (tmp / "src").mkdir()
    (tmp / "src" / "a.py").write_text("x = 1\n")
    (tmp / "src" / "b.py").write_text("y = 2\n")
    (tmp / "config.yaml").write_text("seed: 1\n")
    (tmp / "docs" / "preregistration").mkdir(parents=True)
    (tmp / "docs" / "preregistration" / "config_preregistered.yaml").write_text("seed: 1\n")
    run = tmp / "results" / "r1"
    run.mkdir(parents=True)
    (tmp / "results" / "LATEST").write_text("r1\n")
    (run / "t.csv").write_text("a\n1\n")
    named = hashlib.sha256()
    for n in ["a.py", "b.py"]:
        named.update(n.encode())
        named.update((tmp / "src" / n).read_bytes())
    (run / "run_manifest.json").write_text(json.dumps({
        "run_id": "r1", "created_utc": "20260101T000000Z", "config_path": "config.yaml",
        "config_sha256": _sha(b"seed: 1\n"), "code_sha256": named.hexdigest(),
        "data_source": "data/cache/daily.parquet", "data_sha256": "0" * 64}))
    up = tmp / "results" / "upgrades" / "u"
    up.mkdir(parents=True)
    (up / "run_manifest.json").write_text(json.dumps({
        "run_id": "u1", "config_sha256": _sha(b"seed: 1\n"), "code_sha256": {"a.py": _sha(b"x = 1\n")},
        "input_sha256": {"results/r1/t.csv": _sha(b"a\n1\n")}}))
    return tmp


def test_verify_match_then_mismatch(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    monkeypatch.setattr(vm, "ROOT", root)
    monkeypatch.setattr(vm, "PREREG", root / "docs" / "preregistration" / "config_preregistered.yaml")
    rows = vm.verify(root / "results")
    st = {(r["check"], r["item"]): r["status"] for r in rows}
    assert st[("config", "config.yaml")] == "match"
    assert st[("code", "src/*.py (names and bytes)")] == "match"
    assert st[("code", "src/a.py")] == "match"
    assert st[("data", "results/r1/t.csv")] == "match"
    assert st[("data", "data/cache/daily.parquet")] == "absent"        # data/ is not in the ZIP
    assert st[("pre-registration", "docs/preregistration/config_preregistered.yaml")] == "match"
    assert vm.main(["--results-dir", str(root / "results")]) == 0

    (root / "src" / "a.py").write_text("x = 2\n")                       # code edited after the run
    rows = vm.verify(root / "results")
    bad = {(r["check"], r["item"]) for r in rows if r["status"] == "MISMATCH"}
    assert bad == {("code", "src/*.py (names and bytes)"), ("code", "src/a.py")}
    assert vm.main(["--results-dir", str(root / "results")]) == 1
