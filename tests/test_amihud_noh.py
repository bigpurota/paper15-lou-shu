"""Known-answer tests for src/amihud_noh.py (upgrade 1: Amihud & Noh 2021)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import amihud_noh as an
from measures import daily_returns
from run import synthetic_daily

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _daily(symbol, dates, close, qv):
    return pd.DataFrame({"symbol": symbol, "date": pd.to_datetime(dates), "close": np.asarray(close, float),
                         "quote_volume": np.asarray(qv, float)})


def _toy_random(seed=0, start="2020-01-01", end="2021-03-31", n_sym=2, gap=True):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_sym):
        dates = pd.date_range(start, end, freq="D")
        keep = rng.random(len(dates)) > 0.05
        if gap and i == 0:                          # two months with no data at all
            keep &= ~((dates >= "2020-05-01") & (dates < "2020-07-01"))
        n = len(dates)
        close = 10 * np.exp(np.cumsum(rng.normal(0, 0.04, n)))
        qv = np.exp(rng.normal(14, 0.8, n))
        out.append(_daily(f"S{i}", dates[keep], close[keep], qv[keep]))
    return daily_returns(pd.concat(out, ignore_index=True))


def _brute(daily, sym, s, window, drop_max, scale=1e6):
    """Direct computation over the valid days of months s-window+1..s."""
    v = daily[(daily["symbol"] == sym) & daily["valid"]]
    v = v[(v["month"] > s - window) & (v["month"] <= s)]
    ar = v["r"].abs().to_numpy()
    dv = v["quote_volume"].to_numpy() / scale
    inv, il = 1 / dv, ar / dv

    def m(x):
        return (x.sum() - x.max()) / (len(x) - 1) if drop_max else x.mean()
    cov = np.mean(ar * dv) - ar.mean() * dv.mean()
    var = np.mean(dv ** 2) - dv.mean() ** 2
    return {"n_days": len(v), "ILLIQ": m(il), "absR": m(ar), "IDVOL": m(inv), "b": cov / var,
            "CV2": var / dv.mean() ** 2}


# ---------------------------------------------------------------- rolling 12-month measures
@pytest.mark.parametrize("drop_max", [True, False])
def test_rolling_measures_match_bruteforce(drop_max):
    d = _toy_random()
    m = an.rolling_an_measures(d, window=3, min_days=40, drop_max=drop_max)
    assert len(m) > 10
    for _, row in m.sample(12, random_state=0).iterrows():
        ref = _brute(d, row["symbol"], row["month"], 3, drop_max)
        assert row["n_days"] == ref["n_days"]
        for k in ["ILLIQ", "absR", "IDVOL", "b", "CV2"]:
            assert row[k] == pytest.approx(ref[k], rel=1e-8), k
        assert row["LSIlliq"] == pytest.approx(ref["absR"] * ref["IDVOL"], rel=1e-10)
        assert row["DIF"] == pytest.approx(np.log(ref["ILLIQ"]) - np.log(ref["absR"]) - np.log(ref["IDVOL"]), abs=1e-10)


def test_drop_max_by_hand():
    # closes give |r| = 0.10, 0.20, 0.10, 0.50 on days 2..5; dvol (millions) 1, 2, 4, 5
    d = daily_returns(_daily("X", pd.date_range("2020-01-01", periods=5), [100, 110, 88, 96.8, 145.2],
                             [9e9, 1e6, 2e6, 4e6, 5e6]))
    m = an.rolling_an_measures(d, window=1, min_days=4, drop_max=True).iloc[0]
    ar, dv = np.array([.1, .2, .1, .5]), np.array([1., 2., 4., 5.])
    assert m["n_days"] == 4
    assert m["absR"] == pytest.approx((ar.sum() - 0.5) / 3)                 # drop the 0.50 day
    assert m["IDVOL"] == pytest.approx((np.sum(1 / dv) - 1.0) / 3)          # drop 1/1
    il = ar / dv                                                            # 0.1, 0.1, 0.025, 0.1
    assert m["ILLIQ"] == pytest.approx((il.sum() - 0.1) / 3)
    # b, CV2 on all four days
    assert m["b"] == pytest.approx(np.polyfit(dv, ar, 1)[0])
    assert m["CV2"] == pytest.approx(dv.var() / dv.mean() ** 2)


def test_dif_is_log_of_one_plus_cov_over_lsilliq_without_drop():
    d = _toy_random(seed=3, n_sym=1, gap=False)
    m = an.rolling_an_measures(d, window=2, min_days=30, drop_max=False)
    expect = np.log1p(m["COV12"] / m["LSIlliq"])
    assert np.allclose(m["DIF"], expect, atol=1e-10)


def test_negative_cov_when_absret_rises_with_volume():
    """Karpoff: |r| increasing in volume -> cov(|r|, 1/dvol) < 0 -> DIF < 0 and b > 0;
    with small CV the Taylor term -b CV^2 approximates the covariance (A&N Appendix A)."""
    rng = np.random.default_rng(5)
    n = 120
    dv = 1e6 * (10 + rng.normal(0, 0.5, n))                     # CV ~ 0.05
    ar = 0.01 + 0.002 * (dv / 1e6 - 10) + rng.normal(0, 1e-4, n)
    sign = np.where(rng.random(n) < 0.5, -1, 1)
    close = 100 * np.cumprod(np.r_[1.0, 1 + sign[1:] * ar[1:]])
    d = daily_returns(_daily("K", pd.date_range("2020-01-01", periods=n), close, dv))
    m = an.rolling_an_measures(d, window=4, min_days=100, drop_max=False).iloc[-1]
    assert m["COV12"] < 0 and m["DIF"] < 0 and m["b"] > 0
    assert m["DIF_approx"] == pytest.approx(m["DIF"], rel=0.05)


def test_min_days_and_calendar_window():
    d = _toy_random(seed=1, n_sym=1)            # S0 has no data in May-June 2020
    m = an.rolling_an_measures(d, window=2, min_days=1)
    months = set(m["month"].astype(str))
    assert "2020-06" not in months              # window May-June is empty
    assert "2020-07" in months                  # window June-July has July days only
    j = m.set_index("month").loc[pd.Period("2020-07", "M")]
    assert j["n_days"] == int(d[(d["month"] == pd.Period("2020-07", "M")) & d["valid"]].shape[0])
    m200 = an.rolling_an_measures(d, window=2, min_days=200)
    assert m200.empty


# ---------------------------------------------------------------- cross-sectional helpers
def test_censor_cs_drops_extremes_per_month():
    df = pd.DataFrame({"month": np.repeat([1, 2], 200), "x": np.r_[np.arange(200.0), np.arange(200.0) * 10]})
    out = an.censor_cs(df, ["x"], "month", (0.01, 0.99))
    for mth, g in out.groupby("month"):
        raw = df[df["month"] == mth]["x"]
        assert g["x"].min() >= raw.quantile(0.01) and g["x"].max() <= raw.quantile(0.99)
    assert len(out) == 2 * 196                  # 0,1 and 198,199 fall outside the linear-interpolated quantiles
    nan = df.assign(x=df["x"].where(df.index != 50))
    assert 50 not in an.censor_cs(nan, ["x"], "month", (0.0, 1.0)).index


def test_cs_fit_known_answer():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    df = pd.DataFrame({"month": 1, "x": x, "y": 2 + 3 * x, "z": 3 * x})
    fit, r2 = an.cs_fit(df, "y", ["x"], "month", intercept=True)
    assert np.allclose(fit, df["y"]) and r2.iloc[0] == pytest.approx(1.0)
    fit0, r20 = an.cs_fit(df, "z", ["x"], "month", intercept=False)
    assert np.allclose(fit0, df["z"]) and r20.iloc[0] == pytest.approx(1.0)
    noisy = df.assign(y=df["y"] + rng.normal(size=50))
    _, r2n = an.cs_fit(noisy, "y", ["x"], "month", intercept=True)
    _, r2u = an.cs_fit(noisy, "y", ["x"], "month", intercept=False)
    assert 0 < r2n.iloc[0] < 1 and r2u.iloc[0] < 1
    res = an.cs_residual_multi(noisy, "y", ["x"], "month")
    assert abs(np.corrcoef(res, x)[0, 1]) < 1e-10 and abs(res.mean()) < 1e-10


def test_lag_merge_shifts_two_months():
    an12 = pd.DataFrame({"symbol": ["A", "A"], "month": pd.PeriodIndex(["2020-01", "2020-02"], freq="M"),
                         "DIF": [-0.1, -0.2]})
    panel = pd.DataFrame({"symbol": "A", "month": pd.PeriodIndex(["2020-03", "2020-04", "2020-05"], freq="M"),
                          "ret": [1., 2., 3.], "ret_adj": [1., 2., 3.], "mom": 0., "ret_1": 0., "ivol": 0.1})
    q = an.lag_merge(an12, panel, 2)
    assert q["month"].astype(str).tolist() == ["2020-03", "2020-04"]
    assert q["DIF"].tolist() == [-0.1, -0.2]
    assert q["s"].astype(str).tolist() == ["2020-01", "2020-02"]


# ---------------------------------------------------------------- FM wiring: known premium
def _toy_panel(premium, seed=0, T=80, N=150):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(T):
        lnA_C, lnab, dif = rng.normal(0, 1.5, N), rng.normal(-3, 0.4, N), rng.normal(-0.25, 0.15, N)
        y = premium * dif + 0.0 * lnA_C + rng.normal(0, 5, N)
        rows.append(pd.DataFrame({"month": t, "lnA_C": lnA_C, "ln_absret": lnab, "DIF": dif, "ret_adj": y,
                                  "mom": rng.normal(0, .3, N), "ret_1": rng.normal(0, .1, N)}))
    return pd.concat(rows, ignore_index=True)


def test_fm_recovers_dif_premium_and_null():
    model = {"AN col3": ["lnA_C", "ln_absret", "DIF"]}
    pos = an.fm_block(_toy_panel(10.0), model, ["ret_adj"], 6, "toy").set_index("var")
    assert pos.loc["DIF", "coef"] == pytest.approx(10.0, abs=1.5) and pos.loc["DIF", "t"] > 3
    null = an.fm_block(_toy_panel(0.0, seed=1), model, ["ret_adj"], 6, "toy").set_index("var")
    assert abs(null.loc["DIF", "t"]) < 3


# ---------------------------------------------------------------- verdict logic
def test_classify_and_supports():
    assert an.classify(2.5, 1.96) == "+" and an.classify(2.5, 3.0) == "0"
    assert an.classify(-3.5, 3.0) == "-" and an.classify(np.nan, 1.96) == "n/a"
    assert an.supports("+", "0", "+") == "A&N"
    assert an.supports("0", "0", "+") == an.LS_NULL
    assert an.supports("-", "-", "+") == "LS"
    assert an.supports("-", "0", "+") == "никто"
    assert an.supports("+", "+", "+") == "оба"
    assert an.supports("0", "-", "+") == "никто"
    # "-/0": LS allow an insignificant or a significantly negative residual coefficient
    assert an.supports("0", "-/0", "+") == an.LS_NULL
    assert an.supports("-", "-/0", "+") == "LS"
    assert an.supports("+", "-/0", "+") == "A&N"


def _vt(ts, ls="0", anp="+"):
    fm = []
    for (tid, q, sample, model, var, l_, a_) in an.VERDICT_TESTS:
        fm.append({"sample": sample, "dep": "ret_adj", "model": model, "var": var, "coef": 1.0,
                   "t": ts.get((tid, sample), 0.0), "n_months": 90})
    prem = pd.DataFrame([{"sample": "monthly", "mean_DIF": -0.2, "t_mean_DIF": -20.0, "n_months": 90}])
    return an.verdict_table(pd.DataFrame(fm), prem, (1.96, 3.0))


def test_verdict_rule():
    v = _vt({})                                                   # everything insignificant
    vc = an.verdict_counts(v, 3.0).iloc[0]
    assert vc["verdict"] == "none" and vc["AN_1.96"] == 0 and vc["LS_null_1.96"] > 0
    assert v.loc[v["test"] == "V0", "supports_1.96"].iloc[0] == "A&N"     # premise row, not counted
    vc = an.verdict_counts(_vt({("V1", "monthly"): 3.5, ("V1", "12m"): 2.2}), 3.0).iloc[0]
    assert vc["verdict"] == "AN" and vc["AN_1.96"] == 2 and vc["AN_3"] == 1
    vc = an.verdict_counts(_vt({("V1", "monthly"): 2.2}), 3.0).iloc[0]
    assert vc["verdict"] == "AN_weak"
    vc = an.verdict_counts(_vt({("V3", "monthly"): -4.0, ("V5", "12m"): -2.5}), 3.0).iloc[0]
    assert vc["verdict"] == "LS" and vc["LS_1.96"] == 2
    vc = an.verdict_counts(_vt({("V6", "monthly"): 5.0}), 3.0).iloc[0]   # non-discriminating test ignored
    assert vc["verdict"] == "none"
    vc = an.verdict_counts(_vt({}), 3.0).iloc[0]          # V3/V5 insignificant: LS not rejected, not "никто"
    assert vc["none_1.96"] == 0 and vc["LS_null_1.96"] == vc["n_tests"]


def test_fm_block_min_obs_drops_small_cross_sections():
    df = _toy_panel(0.0, T=10, N=40)
    df = df[~((df["month"] < 3) & (df.index % 40 >= 20))]       # first 3 months: 20 coins only
    model = {"m": ["lnA_C", "ln_absret", "DIF"]}
    a = an.fm_block(df, model, ["ret_adj"], 2, "toy").iloc[0]
    b = an.fm_block(df, model, ["ret_adj"], 2, "toy", min_obs=30).iloc[0]
    assert (a["n_months"], a["min_n"]) == (10, 20) and (b["n_months"], b["min_n"]) == (7, 40)


# ---------------------------------------------------------------- end to end on synthetic data
def _py(script, *args, cwd):
    r = subprocess.run([sys.executable, "-I", str(SRC / script), *args], cwd=cwd,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout}\n{r.stderr}"
    return r


def test_entry_script_end_to_end(tmp_path):
    daily = synthetic_daily(n_symbols=60, start="2018-01-01", end="2020-06-30", seed=1)
    daily_path = tmp_path / "daily.parquet"
    daily.to_parquet(daily_path, index=False)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["period"] = {"return_start": "2019-01", "return_end": "2020-06"}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    res, cache, out = tmp_path / "results", tmp_path / "cache", tmp_path / "an"
    _py("run.py", str(cfg_path), "--daily", str(daily_path), "--results-dir", str(res),
        "--cache-dir", str(cache), "--no-placebo", cwd=tmp_path)
    base_id = (res / "LATEST").read_text().strip()

    _py("amihud_noh.py", str(cfg_path), "--results-dir", str(res), "--cache-dir", str(cache),
        "--out", str(out), cwd=tmp_path)
    man = json.loads((out / "run_manifest.json").read_text())
    assert man["base_run_id"] == base_id and man["data_matches_base"] is True
    for k in ["run_id", "git", "config_sha256", "code_sha256", "inputs", "seed", "timings_sec", "spec"]:
        assert k in man, k
    assert len(man["inputs"]["daily"]["sha256"]) == 64 and "amihud_noh.py" in man["code_sha256"]
    for t in man["tables"]:
        assert (out / t).exists(), t
    v = pd.read_csv(out / "verdict.csv")
    assert {"V0", "V1", "V2", "V3", "V4", "V5", "V6"} <= set(v["test"])
    assert set(v["dep"]) == {"-", "ret_adj", "ret"}
    fm = pd.read_csv(out / "fm.csv")
    assert {"monthly", "12m"} == set(fm["sample"])
    d = pd.read_csv(out / "describe.csv").set_index("sample")
    assert d.loc["12m", "n_coin_months_in_return_panel"] > 0
    s = (out / "SUMMARY.md").read_text()
    assert "Вердикт" in s and base_id in s and "nan" not in s.lower().replace("n/a", "")

    # a changed daily file must stop the run unless explicitly allowed
    daily.assign(close=daily["close"] * 1.01).to_parquet(daily_path, index=False)
    r = subprocess.run([sys.executable, "-I", str(SRC / "amihud_noh.py"), str(cfg_path), "--results-dir", str(res),
                        "--cache-dir", str(cache), "--out", str(out)], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode != 0 and "differs from base run" in r.stderr
