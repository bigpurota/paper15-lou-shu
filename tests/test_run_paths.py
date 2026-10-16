"""rel_path keeps repo paths repo-relative, also when data/ is a symlink to a directory outside the repo."""
import os

import run


def test_rel_path_inside_repo():
    assert run.rel_path(run.ROOT / "data" / "cache" / "daily.parquet") == "data/cache/daily.parquet"


def test_rel_path_through_symlinked_dir(tmp_path):
    outside = tmp_path / "outside_data"
    (outside / "cache").mkdir(parents=True)
    (outside / "cache" / "x.parquet").write_bytes(b"")
    link = tmp_path / "repo_like"
    link.mkdir()
    os.symlink(outside, link / "data")
    old_root = run.ROOT
    try:
        run.ROOT = link.resolve()
        out = run.rel_path(run.ROOT / "data" / "cache" / "x.parquet")
    finally:
        run.ROOT = old_root
    assert out == "data/cache/x.parquet"
    assert "/Users/" not in out and not out.startswith("..")
