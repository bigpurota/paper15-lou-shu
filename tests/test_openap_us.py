"""Known-answer tests for the OpenAP (US) upgrade: parsing, NW regressions, spanning, caching, end-to-end."""
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

import openap_us as oa

FF_TEXT = """This file was created using the 202608 CRSP database.
Some header text.

,Mkt-RF,SMB,HML,RF
196401,   1.00,   2.00,  -99.99,   0.30
196402,   0.50,  -1.00,   1.50,   0.25

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
1964,  10.00,   1.00,   2.00,   3.00
"""
MOM_TEXT = """Momentum header

Missing data are indicated by -99.99 or -999.

,Mom
196401,   0.70
196402,  -0.20

 Annual Factors:
,Mom
1964,   5.00
"""


def test_parse_french_monthly_block_only():
    df = oa.parse_french_csv(FF_TEXT)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(df) == 2
    assert df.index[0] == pd.Period("1964-01", "M")
    assert np.isnan(df.loc[pd.Period("1964-01", "M"), "HML"])
    assert df.loc[pd.Period("1964-02", "M"), "SMB"] == -1.0


def _write_zip(path, name, text):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(name, text)


def test_load_french_joins_umd(tmp_path):
    _write_zip(tmp_path / "ff.zip", "ff.csv", FF_TEXT)
    _write_zip(tmp_path / "mom.zip", "mom.csv", MOM_TEXT)
    df = oa.load_french(tmp_path / "ff.zip", tmp_path / "mom.zip")
    assert "UMD" in df.columns and df["UMD"].tolist() == [0.7, -0.2]


def test_load_ls_wide(tmp_path):
    p = tmp_path / "w.csv"
    p.write_text("date,Illiquidity,DolVol\n1964-01-31,1.5,\n1964-02-29,-0.5,2.0\n")
    df = oa.load_ls_wide(p)
    assert df.index.name == "month" and df.index[1] == pd.Period("1964-02", "M")
    assert np.isnan(df.iloc[0]["DolVol"]) and df.iloc[1]["DolVol"] == 2.0


def _months(n, start="1960-01"):
    return pd.period_range(start, periods=n, freq="M", name="month")


def test_ols_hac_known_answer():
    rng = np.random.default_rng(0)
    idx = _months(400)
    x = pd.Series(rng.normal(0, 1, 400), idx)
    y = 0.5 + 2.0 * x + rng.normal(0, 0.01, 400)
    r = oa.ols_hac(y, x.to_frame("x"), lags=6)
    assert r["alpha"] == pytest.approx(0.5, abs=0.005)
    assert r["x"] == pytest.approx(2.0, abs=0.005)
    assert r["n"] == 400 and r["r2"] > 0.99
    m = oa.ols_hac(pd.Series(np.r_[np.ones(10), 3 * np.ones(10)], _months(20)), None, lags=6)
    assert m["alpha"] == pytest.approx(2.0)


def test_mean_tests_difference_known_answer():
    rng = np.random.default_rng(1)
    idx = pd.period_range("1964-01", "2024-12", freq="M", name="month")
    pre = idx <= pd.Period("2012-12", "M")
    s = pd.Series(np.where(pre, 1.0, 0.2) + rng.normal(0, 0.05, len(idx)), idx)
    out = oa.mean_tests(s.to_frame("Illiquidity"), oa.PERIODS, lags=6).set_index("period")
    assert out.loc["LS_1964_2012", "n"] == 588
    assert out.loc["post_2013", "n"] == 144
    assert out.loc["LS_1964_2012", "mean"] == pytest.approx(1.0, abs=0.02)
    assert out.loc["post_2013_minus_LS_1964_2012", "mean"] == pytest.approx(-0.8, abs=0.03)
    assert out.loc["post_2013_minus_LS_1964_2012", "t"] < -10


def _toy_world(seed=2, alpha=0.3, beta=0.5):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1964-01", "2024-12", freq="M", name="month")
    n = len(idx)
    ff = pd.DataFrame(rng.normal(0, 3, (n, 4)), idx, columns=["Mkt-RF", "SMB", "HML", "UMD"])
    dv = pd.Series(0.6 + 0.8 * ff["SMB"] + rng.normal(0, 1, n), idx)
    il = alpha + beta * dv + rng.normal(0, 0.2, n)
    return pd.DataFrame({"Illiquidity": il, "DolVol": dv}), ff


def test_spanning_recovers_alpha_and_beta():
    ls, ff = _toy_world(alpha=0.3, beta=0.5)
    sp = oa.spanning_tests(ls, ff, "Illiquidity", "DolVol", oa.PERIODS, lags=6).set_index(["period", "spec"])
    for p in oa.PERIODS:
        assert sp.loc[(p, "one_factor"), "alpha"] == pytest.approx(0.3, abs=0.05)
        assert sp.loc[(p, "one_factor"), "b_factor"] == pytest.approx(0.5, abs=0.02)
        assert sp.loc[(p, "five_factor"), "b_factor"] == pytest.approx(0.5, abs=0.02)
        assert sp.loc[(p, "five_factor"), "alpha"] == pytest.approx(0.3, abs=0.05)
    ls0, ff0 = _toy_world(seed=3, alpha=0.0, beta=0.5)
    sp0 = oa.spanning_tests(ls0, ff0, "Illiquidity", "DolVol", oa.PERIODS, lags=6).set_index(["period", "spec"])
    assert abs(sp0.loc[("LS_1964_2012", "one_factor"), "t_alpha"]) < 3


def test_correlation_and_sign_checks():
    ls, ff = _toy_world()
    c = oa.correlations(ls, "Illiquidity", "DolVol", oa.PERIODS)
    assert set(c["period"]) == {"full", *oa.PERIODS} and (c["pearson"] > 0.9).all()
    doc = pd.DataFrame({"Acronym": ["Illiquidity", "DolVol"], "Sign": [1.0, -1.0]})
    s = oa.sign_checks(ls, ff, doc, ["Illiquidity", "DolVol"], lags=6).set_index("signal")
    assert s.loc["Illiquidity", "long_side"].startswith("top")
    assert s.loc["DolVol", "long_side"].startswith("bottom")
    assert s["smb_check_pass"].all()


def test_fetch_falls_back_on_quota_page_and_caches(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, timeout=300):
        calls.append(url)
        if "drive" in url:
            return b"<!DOCTYPE html><html><head><title>Google Drive - Quota exceeded</title></head></html>"
        return b"date,Illiquidity\n1964-01-31,1.0\n"

    monkeypatch.setattr(oa, "_get", fake_get)
    dest = tmp_path / "x" / "f.csv"
    rec = oa.fetch(dest, ["https://drive.example/a", "https://mirror.example/b"])
    assert rec["url"] == "https://mirror.example/b" and "Quota exceeded" in rec["tried_before"][0]
    assert dest.read_bytes().startswith(b"date")
    monkeypatch.setattr(oa, "_get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    rec2 = oa.fetch(dest, ["https://drive.example/a"])
    assert rec2["cached"] and rec2["url"] == "https://mirror.example/b" and rec2["sha256"] == rec["sha256"]


def test_fetch_raises_when_all_sources_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(oa, "_get", lambda *a, **k: b"<html>nope</html>")
    with pytest.raises(RuntimeError, match="could not download"):
        oa.fetch(tmp_path / "f.csv", ["https://a.example", "https://b.example"])


def test_main_end_to_end_offline(tmp_path, monkeypatch):
    """Full entry point on a toy cached raw dir: reads LATEST, writes CSVs, SUMMARY.md, manifest."""
    monkeypatch.setattr(oa, "_get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    raw = tmp_path / "raw"
    od = raw / "openap" / f"release_{oa.OPENAP_RELEASE}"
    od.mkdir(parents=True)
    ls, ff = _toy_world()
    w = ls.copy()
    w.insert(0, "date", w.index.to_timestamp(how="end").strftime("%Y-%m-%d"))
    w.to_csv(od / "PredictorLSretWide.csv", index=False)
    pd.DataFrame({"Acronym": ["Illiquidity", "DolVol"], "Sign": [1, -1], "Stock Weight": ["EW", "EW"],
                  "Portfolio Period": [12, 1], "Filter": ["exchcd==1", np.nan], "LS Quantile": [np.nan, np.nan],
                  "Detailed Definition": ["toy A", "toy DV"]}).to_csv(od / "SignalDoc.csv", index=False)
    (raw / "french").mkdir()
    ffm = ff.copy()
    ffm["RF"] = 0.3
    ff_txt = "hdr\n\n," + ",".join(["Mkt-RF", "SMB", "HML", "RF"]) + "\n" + "\n".join(
        f"{p.strftime('%Y%m')},{r['Mkt-RF']:.4f},{r['SMB']:.4f},{r['HML']:.4f},{r['RF']:.4f}" for p, r in ffm.iterrows())
    mom_txt = "hdr\n\n,Mom\n" + "\n".join(f"{p.strftime('%Y%m')},{v:.4f}" for p, v in ff["UMD"].items())
    _write_zip(raw / "french" / oa.FRENCH_FILES[0], "a.csv", ff_txt + "\n\n Annual\n")
    _write_zip(raw / "french" / oa.FRENCH_FILES[1], "b.csv", mom_txt + "\n")
    res = tmp_path / "results"
    res.mkdir()
    (res / "LATEST").write_text("TOYRUN_123\n")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("seed: 1\ntests:\n  nw_lags: 6\n  hlz_threshold: 3.0\n")
    out = oa.main([str(cfg), "--results-dir", str(res), "--out", str(tmp_path / "out"), "--raw", str(raw)])
    man = json.loads((out / "run_manifest.json").read_text())
    assert man["base_run_id"] == "TOYRUN_123" and man["nw_lags"] == 6
    for f in man["tables"]:
        assert (out / f).exists()
    summ = (out / "SUMMARY.md").read_text()
    assert "TOYRUN_123" in summ and "nan" not in summ.lower()
    pc = pd.read_csv(out / "paper_comparison.csv").set_index("item")
    assert pc.loc["TableA1_one_factor_alpha", "ours"] == pytest.approx(0.3, abs=0.06)


def test_decay_lag_sensitivity_and_correlation_span():
    ls, _ = _toy_world()
    s = oa.decay_lag_sensitivity(ls, oa.PERIODS, lags_list=(0, 6), primary=6)
    assert set(s["nw_lags"]) == {0, 6} and s.loc[s["nw_lags"] == 6, "primary"].all()
    m6 = oa.mean_tests(ls, oa.PERIODS, 6).set_index(["signal", "period"])
    r = s[(s["signal"] == "DolVol") & (s["nw_lags"] == 6)].iloc[0]
    assert r["t"] == pytest.approx(m6.loc[("DolVol", r["period"]), "t"])
    c = oa.correlations(ls, "Illiquidity", "DolVol", oa.PERIODS).set_index("period")
    assert c.loc["LS_1964_2012", "start"] == "1964-01" and c.loc["LS_1964_2012", "end"] == "2012-12"


def test_signal_meta_effective_quantile_default():
    doc = pd.DataFrame({"Acronym": ["Illiquidity", "zerotrade1M"], "LS Quantile": [np.nan, 0.1]})
    m = oa.signal_meta(doc, ["Illiquidity", "zerotrade1M"]).set_index("Acronym")
    assert m.loc["Illiquidity", "LS Quantile effective"] == oa.OPENAP_DEFAULT_LS_QUANTILE == 0.2
    assert m.loc["zerotrade1M", "LS Quantile effective"] == 0.1
