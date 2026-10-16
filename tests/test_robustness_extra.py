"""Known-answer tests for the robustness upgrade: block bootstrap, MDE, ledger, stability, end to end."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

import robustness_extra as rx
from fama_macbeth import nw_mean
from run import synthetic_daily

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


# ---------------------------------------------------------------- moving-block bootstrap
def test_mbb_indices_shape_range_and_blocks():
    rng = np.random.default_rng(0)
    idx = rx.mbb_indices(93, 6, 500, rng)
    assert idx.shape == (500, 93)
    assert idx.min() >= 0 and idx.max() <= 92
    # inside each block of 6 the indices are consecutive
    d = np.diff(idx[:, :90].reshape(500, 15, 6), axis=2)
    assert (d == 1).all()
    # deterministic given the seed
    assert (rx.mbb_indices(93, 6, 500, np.random.default_rng(0)) == idx).all()
    with pytest.raises(ValueError):
        rx.mbb_indices(5, 6, 10, rng)


def test_mbb_block1_iid_se_matches_formula():
    rng = np.random.default_rng(1)
    x = rng.normal(0.5, 2.0, 400)
    draws = rx.boot_means(x, rx.mbb_indices(len(x), 1, 4000, np.random.default_rng(2)))
    assert draws.mean() == pytest.approx(x.mean(), abs=0.02)
    assert draws.std(ddof=1) == pytest.approx(x.std(ddof=0) / np.sqrt(len(x)), rel=0.06)


def test_mbb_captures_autocorrelation():
    """AR(1), phi = 0.6: MBB variance with block b ~ Bartlett-weighted long-run variance,
    sum_{|k|<b} (1-|k|/b) phi^|k| times the marginal variance (= iid bootstrap, block 1)."""
    rng = np.random.default_rng(3)
    e = rng.normal(size=3000)
    x = np.zeros_like(e)
    for i in range(1, len(e)):
        x[i] = 0.6 * x[i - 1] + e[i]
    se1 = rx.boot_means(x, rx.mbb_indices(len(x), 1, 2000, np.random.default_rng(4))).std()
    se6 = rx.boot_means(x, rx.mbb_indices(len(x), 6, 2000, np.random.default_rng(4))).std()
    se30 = rx.boot_means(x, rx.mbb_indices(len(x), 30, 2000, np.random.default_rng(4))).std()
    assert se1 < se6 < se30
    for b, se in [(6, se6), (30, se30)]:
        k = np.arange(1, b)
        expected = np.sqrt(1 + 2 * np.sum((1 - k / b) * 0.6 ** k))
        assert se / se1 == pytest.approx(expected, rel=0.1)


def test_boot_means_ignores_nan_months():
    x = np.array([1.0, np.nan, 3.0, np.nan])
    idx = np.array([[0, 1, 2, 3], [1, 1, 0, 0]])
    assert rx.boot_means(x, idx) == pytest.approx([2.0, 1.0])


def test_boot_alphas_recovers_intercept():
    rng = np.random.default_rng(5)
    F = rng.normal(size=(120, 2))
    y = 0.7 + F @ np.array([1.5, -0.5]) + 0.01 * rng.normal(size=120)
    a = rx.boot_alphas(y, F, rx.mbb_indices(120, 6, 200, rng))
    assert np.allclose(a, 0.7, atol=0.01)


def test_ci_row_coverage_iid():
    """Percentile CI of the mean covers the true mean ~95% of the time (iid, T = 93)."""
    rng = np.random.default_rng(6)
    hits = 0
    n = 300
    for _ in range(n):
        x = rng.normal(0.3, 1.0, 93)
        r = rx.ci_row("x", x.mean(), np.nan, rx.boot_means(x, rx.mbb_indices(93, 6, 400, rng)), 93)
        hits += r["ci95_lo"] <= 0.3 <= r["ci95_hi"]
    assert 0.88 <= hits / n <= 0.99


def test_bootstrap_seed_stability_flags_borderline():
    rng = np.random.default_rng(3)
    T = 90
    idx = pd.period_range("2019-01", periods=T, freq="M")
    S = pd.DataFrame({"hl_far": rng.normal(3.0, 1.0, T),          # mean far from 0
                      "fm_zero": rng.normal(0.0, 1.0, T)}, index=idx)
    # a series whose mean sits exactly at the 2.5% bootstrap quantile edge: lower bound ~ 0
    z = rng.normal(0.0, 1.0, T)
    S["fm_edge"] = z - z.mean() + 1.96 * z.std(ddof=1) / np.sqrt(T)
    F = pd.DataFrame({"CMKT": rng.normal(size=T), "CMOM": rng.normal(size=T)}, index=idx)
    B = rx.bootstrap_table(S, F, lags=0, block=1, n_draws=1000, seed=1, n_seed_reps=20).set_index("stat")
    assert B.loc["hl_far", "ci_verdict"] == "excludes 0 (all seeds)"
    assert B.loc["fm_zero", "ci_verdict"] == "contains 0 (all seeds)"
    assert B.loc["fm_edge", "ci_verdict"] == "borderline (depends on seed)"
    assert 0 < B.loc["fm_edge", "share_seeds_ci_excludes_0"] < 1
    assert (B["ci95_lo_seed_min"] <= B["ci95_lo_seed_max"]).all()


def test_ci_row_flags():
    r = rx.ci_row("x", 1.0, 3.0, np.r_[np.linspace(0.5, 1.5, 999), np.nan], 50)
    assert r["ci_excludes_0"] and r["share_draws_opposite_sign"] == 0.0 and r["n_draws_valid"] == 999
    r = rx.ci_row("x", 1.0, 0.5, np.linspace(-1, 3, 1000), 50)
    assert not r["ci_excludes_0"] and r["share_draws_opposite_sign"] == pytest.approx(0.25, abs=0.01)


# ---------------------------------------------------------------- stability
def _toy_series(seed=7):
    rng = np.random.default_rng(seed)
    months = pd.period_range("2019-01", "2022-12", freq="M")
    S = pd.DataFrame({"hl_A_C": rng.normal(0.2, 1.0, len(months)),
                      "fm2_lnA_C": rng.normal(0.1, 0.5, len(months))}, index=months)
    S.loc[[p.year == 2021 for p in months], "hl_A_C"] += 10.0         # one huge year
    F = pd.DataFrame({"CMKT": rng.normal(size=len(months)), "CMOM": rng.normal(size=len(months))}, index=months)
    return S, F


def test_loyo_drops_one_year():
    S, F = _toy_series()
    L = rx.loyo_table(S, F, 6)
    assert set(L["dropped_year"]) == {"none", "2019", "2020", "2021", "2022"}
    full = L[(L["dropped_year"] == "none") & (L["stat"] == "hl_A_C")].iloc[0]
    m, _, t = nw_mean(S["hl_A_C"], 6)
    assert full["estimate"] == pytest.approx(m) and full["t_nw"] == pytest.approx(t) and full["n_months"] == 48
    d21 = L[(L["dropped_year"] == "2021") & (L["stat"] == "hl_A_C")].iloc[0]
    assert d21["n_months"] == 36
    assert d21["estimate"] == pytest.approx(S.loc[[p.year != 2021 for p in S.index], "hl_A_C"].mean())
    summ = rx.loyo_summary(L).set_index("stat")
    assert summ.loc["hl_A_C", "year_t_min"] == "2021" and summ.loc["hl_A_C", "n_years"] == 4
    ex = rx.excl_year_table(S.assign(hl_A=0.0 + S["hl_A_C"], hl_ResA=S["hl_A_C"], fm1_lnA=S["fm2_lnA_C"],
                                     fm3_ResLnA=S["fm2_lnA_C"], fm4_lnA_C=S["fm2_lnA_C"], fm4_ResLnA=S["fm2_lnA_C"]),
                            F, 6, 2021)
    r = ex[(ex["stat"] == "hl_A_C")].set_index("sample")
    assert r.loc["excl_2021", "estimate"] == pytest.approx(d21["estimate"])


# ---------------------------------------------------------------- simulation calibration and MDE
def test_world0_calibration_standard_normal():
    rng = np.random.default_rng(8)
    w0 = pd.DataFrame({"fm2_lnA_C_t": rng.normal(size=200000), "fm2_lnA_C_nw0_t": rng.normal(size=200000)})
    c = rx.world0_calibration(w0, 3.0).set_index("stat")
    assert list(c.index) == ["fm2_lnA_C"]
    assert c.loc["fm2_lnA_C", "fpr_1.96"] == pytest.approx(0.05, abs=0.003)
    assert c.loc["fm2_lnA_C", "fpr_3"] == pytest.approx(2 * norm.sf(3), abs=0.001)
    assert c.loc["fm2_lnA_C", "crit_abs_t_5pct"] == pytest.approx(1.96, abs=0.02)


def test_mde_from_power_curve_interpolates_and_monotonises():
    prem = np.array([0.0, 0.25, 0.5, 1.0])
    assert rx.mde_from_power_curve(prem, np.array([0.05, 0.35, 0.95, 1.0])) == pytest.approx(0.25 + 0.45 / 0.6 * 0.25)
    # non-monotone noise: running max is used
    assert rx.mde_from_power_curve(prem, np.array([0.05, 0.6, 0.5, 1.0])) == pytest.approx(0.5 + 0.2 / 0.4 * 0.5)
    assert np.isnan(rx.mde_from_power_curve(prem, np.array([0.05, 0.1, 0.2, 0.3])))


def test_mde_from_mean_t_known_answer():
    prem = np.array([0.0, 0.25, 0.5, 1.0])
    mean_t = 7.5 * prem                       # E[t] linear, through 0
    assert rx.mde_from_mean_t(prem, mean_t) == pytest.approx((1.96 + norm.ppf(0.8)) / 7.5)


def test_mde_table_real_se_and_scaling():
    power = pd.DataFrame({"world": "V", "stat": "fm2_lnA_C", "premium": [0.0, 0.0025, 0.005, 0.01],
                          "n_reps": 20, "mean_t": [0.0, 1.5, 3.0, 6.0],
                          "share_abs_t_gt_1.96": [0.05, 0.35, 0.95, 1.0],
                          "ci95_lo": [0.0, 0.15, 0.75, 0.83], "ci95_hi": [0.3, 0.6, 1.0, 1.0]})
    w0 = pd.DataFrame({"stat": ["fm2_lnA_C"], "crit_abs_t_5pct": [2.2]})
    rse = pd.DataFrame([{"stat": "fm2_lnA_C", "se_nw": 0.1, "se_boot": 0.12, "sd_signal": 2.0,
                         "units": "u", "crit_world0": 2.2}])
    M = rx.mde_table(power, w0, {"N": 400, "T": 100}, {"N": 100, "T": 100}, rse).set_index(["stat", "method"])
    z = norm.ppf(0.8)
    assert M.loc[("fm2_lnA_C", "real_se_nw_crit_1.96"), "mde"] == pytest.approx((1.96 + z) * 0.1 * 2.0)
    assert M.loc[("fm2_lnA_C", "real_se_boot_crit_world0_5pct"), "mde"] == pytest.approx((2.2 + z) * 0.12 * 2.0)
    base = M.loc[("fm2_lnA_C", "sim_mean_t_crit_1.96"), "mde"]
    assert base == pytest.approx((1.96 + z) / 6.0 * 1.0)          # mean_t = 600 * premium(fraction) = 6 per %
    assert M.loc[("fm2_lnA_C", "sim_scaled_to_real_crit_1.96"), "mde"] == pytest.approx(2.0 * base)
    assert M.loc[("fm2_lnA_C", "sim_power_curve"), "mde"] == pytest.approx(100 * (0.0025 + 0.45 / 0.6 * 0.0025))


def test_paper_effect_rows_from_yaml():
    paper = yaml.safe_load((SRC / "paper_numbers.yaml").read_text())
    P = rx.paper_effect_rows(paper).set_index("stat")
    # the paper states the s.d. of ln(A_C) (2.53) and the one-s.d. effect (0.46%/mo) on p. 13
    assert P.loc["fm2_lnA_C", "sd_signal"] == pytest.approx(2.53)
    assert P.loc["fm2_lnA_C", "mde"] == pytest.approx(0.183 * 2.53, rel=1e-6)
    assert round(P.loc["fm2_lnA_C", "mde"], 2) == 0.46
    q = paper["lou_shu_2017"]["table1_panelA"]["A_C"]
    assert P.loc["fm2_lnA_C", "sd_signal_iqr_approx"] == pytest.approx(np.log(q["q75"] / q["q25"]) / 1.3489795)
    assert P.loc["hl_A_C", "mde"] == paper["lou_shu_2017"]["table3_panelA"]["A_C"]["raw"]["hl"]


# ---------------------------------------------------------------- ledger
def _write(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_ledger_counts_flags_duplicates_roles(tmp_path):
    base = tmp_path / "results" / "run1"
    _write(base / "T4_fm.csv", pd.DataFrame({"model": ["(1)", "(1)", "(2)"], "var": ["lnA", "mom", "lnA_C"],
                                             "coef": [0.1, 0.2, 0.3], "t": [2.5, 1.0, 3.5], "n_months": [90] * 3}))
    _write(base / "T2_fm.csv", pd.DataFrame({"model": ["(1)"], "var": ["lnA"], "coef": [0.1], "t": [2.5]}))  # dup
    _write(base / "T3_sorts.csv", pd.DataFrame({"row": ["raw"], "signal": ["A"], "Q1": [1.0], "H-L": [0.6],
                                                "t": [-1.0], "n_months": [93]}))
    _write(base / "placebo.csv", pd.DataFrame({"test": ["shuffle"], "var": ["x"], "coef": [0.0], "t": [2.1],
                                               "sd_t": [1.0], "share_abs_t_gt_1.96": [0.05]}))
    _write(base / "paper_vs_us.csv", pd.DataFrame({"id": ["a"], "paper": [1.0], "paper_t": [5.0],
                                                   "us": [1.0], "us_t": [9.0]}))
    _write(base / "T1_corr.csv", pd.DataFrame({"measure": ["A"], "A": [1.0]}))                 # no t
    upg = tmp_path / "results" / "upgrades"
    _write(upg / "vw" / "fm.csv", pd.DataFrame({"spec": ["vw"], "estimate": [0.4], "t_nw": [-3.2]}))
    _write(upg / "robustness" / "ledger.csv", pd.DataFrame({"estimate": [9.0], "t": [99.0]}))  # own, excluded
    _write(upg / "robustness" / "stability_loyo.csv",
           pd.DataFrame({"dropped_year": ["2021"], "stat": ["hl_A_C"], "estimate": [0.2], "t_nw": [0.5]}))

    L = rx.build_ledger(base, upg, exclude=rx.OWN_NON_TESTS, hlz=3.0)
    assert set(L["file"]) == {"T4_fm.csv", "T2_fm.csv", "T3_sorts.csv", "placebo.csv", "fm.csv", "stability_loyo.csv"}
    assert len(L) == 8
    lnA = L[L["test"] == "model=(1) | var=lnA"]              # T2 is read first (sorted), T4's copy is the dup
    assert list(lnA["unique"]) == [True, False]
    assert lnA["duplicate_of"].iloc[1].startswith("base/T2_fm.csv")
    assert L.loc[L["file"] == "T3_sorts.csv", "estimate"].iloc[0] == 0.6          # H-L picked as estimate
    assert L.loc[L["file"] == "fm.csv", "estimate"].iloc[0] == 0.4
    roles = dict(zip(L["file"] + ":" + L["test"], L["role"]))
    assert roles["T4_fm.csv:model=(1) | var=mom"] == "control"
    assert L.loc[L["file"] == "placebo.csv", "role"].iloc[0] == "placebo"
    assert L.loc[L["file"] == "fm.csv", "source"].iloc[0] == "upgrade:vw"
    assert L["abs_t_gt_3"].sum() == 2                     # 3.5 and -3.2
    L = rx.add_multiple_testing(L)
    fam = L[L["in_family"]]
    # family: T4 lnA, T4 lnA_C, T3 H-L, vw, robustness stability (T2 dup, mom control, placebo excluded)
    assert len(fam) == 5 and (L["family_size"] == 5).all()
    p = 2 * norm.sf(fam["t"].abs().to_numpy())
    assert np.allclose(fam["q_bh"], multipletests(p, method="fdr_bh")[1])
    assert np.allclose(fam["p_bonferroni"], np.minimum(1, 5 * p))
    S = rx.ledger_summary(L, 3.0).set_index(["source", "role"])
    a = S.loc[("ALL", "test (family)")]
    assert a["n_unique"] == 5 and a["n_abs_t_gt_1.96"] == 3 and a["n_abs_t_gt_3"] == 2
    assert a["bonferroni_crit_t"] == pytest.approx(norm.isf(0.05 / 10))


def test_t_column_detection():
    df = pd.DataFrame({"name": ["a"], "coef": [1.0], "t": [2.0], "sd_t": [1.0], "paper_t": [3.0],
                       "share_t_gt": [0.1], "beta_t": [4.0], "beta": [0.5]})
    assert rx._t_columns(df) == ["t", "beta_t"]
    assert rx._estimate_col(df, "beta_t") == "beta"
    assert rx._estimate_col(df, "t") == "coef"
    # time-indexed quantities are not t-statistics
    df2 = pd.DataFrame({"mean_dvol_t": [1e7], "dvol_t": [1e6], "max_t": [0.5], "t": [1.0]})
    assert rx._t_columns(df2) == ["t"]
    # a text column named like the t-column's prefix is never the estimate
    df3 = pd.DataFrame({"statistic": ["HL A"], "variant": ["winsorized"], "base": [1.5], "base_t": [0.98],
                        "variant_coef": [1.3], "variant_t": [0.81]})
    assert rx._estimate_col(df3, "variant_t") != "variant"
    assert rx._estimate_col(df3, "base_t") == "base"


def test_ledger_roles_diagnostics_loadings_legs(tmp_path):
    base = tmp_path / "results" / "run1"
    _write(base / "T4_fm.csv", pd.DataFrame({"var": ["lnA"], "coef": [0.1], "t": [1.0]}))
    _write(base / "ret_extremes.csv", pd.DataFrame({"symbol": ["X"], "mean_dvol_t": [1e7], "t": [5.0]}))
    _write(base / "audit_summary.csv", pd.DataFrame({"type": ["x"], "t": [5.0]}))
    _write(base / "robustness_vs_base.csv", pd.DataFrame({"statistic": ["HL A"], "variant": ["w"],
                                                          "base": [1.5], "base_t": [0.98], "variant_t": [0.8]}))
    _write(base / "placebo.csv", pd.DataFrame({"test": ["leak (lag 0)", "shuffle within month"],
                                               "var": ["H-L A", "H-L A"], "coef": [5.0, 0.0], "t": [6.0, 0.1]}))
    upg = tmp_path / "results" / "upgrades"
    _write(upg / "us" / "spanning.csv", pd.DataFrame({"dep": ["I"], "alpha": [0.1], "t_alpha": [0.6],
                                                      "b_SMB": [0.5], "t_SMB": [15.0]}))
    _write(upg / "s" / "panel_a.csv", pd.DataFrame({"state": ["hi"], "Short": [4.0], "t_Short": [0.8],
                                                    "Long-Short": [3.5], "t_Long-Short": [1.4]}))
    _write(upg / "s" / "panel_b.csv", pd.DataFrame({"portfolio": ["2", "Long-Short"], "b": [0.1, 0.2],
                                                    "t": [0.4, 1.1], "a": [0.0, 0.0], "t_a": [0.0, 0.1]}))
    _write(upg / "hf" / "lambda_monthly.csv", pd.DataFrame({"symbol": ["X"], "lam": [1e-5], "t_lam": [30.0]}))
    _write(upg / "an" / "describe.csv", pd.DataFrame({"sample": ["m"], "mean_DIF": [-0.2], "t_mean_DIF": [-27.0]}))
    _write(upg / "an" / "verdict.csv", pd.DataFrame({"test": ["V0"], "coef": [-0.2], "t": [-27.0]}))  # dup
    _write(upg / "an" / "paper_reference.csv", pd.DataFrame({"var": ["lnA_C"], "coef": [0.1], "t": [3.9]}))
    L = rx.build_ledger(base, upg, hlz=3.0)
    assert set(L["file"]) == {"T4_fm.csv", "placebo.csv", "spanning.csv", "panel_a.csv", "panel_b.csv",
                              "lambda_monthly.csv", "describe.csv", "verdict.csv"}
    assert L.loc[L["file"] == "lambda_monthly.csv", "role"].iloc[0] == "measurement"
    assert L.loc[L["file"] == "describe.csv", "role"].iloc[0] == "descriptive"
    assert not L.loc[L["file"] == "verdict.csv", "unique"].iloc[0]
    r = dict(zip(L["file"] + ":" + L["t_column"] + ":" + L["row"].astype(str), L["role"]))
    assert r["placebo.csv:t:0"] == "diagnostic" and r["placebo.csv:t:1"] == "placebo"
    assert r["spanning.csv:t_alpha:0"] == "test" and r["spanning.csv:t_SMB:0"] == "control"
    assert r["panel_a.csv:t_Short:0"] == "descriptive" and r["panel_a.csv:t_Long-Short:0"] == "test"
    assert r["panel_b.csv:t:0"] == "descriptive" and r["panel_b.csv:t:1"] == "test"
    assert r["panel_b.csv:t_a:1"] == "control"


# ---------------------------------------------------------------- end to end on a synthetic base run
def _py(script, *args, cwd):
    r = subprocess.run([sys.executable, "-I", str(SRC / script), *args], cwd=cwd,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout}\n{r.stderr}"
    return r


def _toy_sims(d: Path, seed=9):
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True)
    cols = ["fm2_lnA_C", "fm3_ResLnA", "fm4_lnA_C", "fm4_ResLnA", "hl_A", "hl_A_C", "hl_ResA"]
    pd.DataFrame({f"{c}_t": rng.normal(size=200) for c in cols}).to_csv(d / "sim_world0_reps.csv", index=False)
    rows = []
    for st in ["fm2_lnA_C", "hl_A_C"]:
        for pr, pw, mt in [(0.0, 0.05, 0.0), (0.0025, 0.4, 1.6), (0.005, 0.95, 3.6), (0.01, 1.0, 7.0)]:
            rows.append({"world": "V", "premium": pr, "stat": st, "n_reps": 20, "mean_t": mt,
                         "share_abs_t_gt_1.96": pw, "ci95_lo": max(0, pw - 0.2), "ci95_hi": min(1, pw + 0.2)})
    pd.DataFrame(rows).to_csv(d / "sim_power.csv", index=False)
    pd.DataFrame({"avg_coins": [300.0] * 4, "n_months": [90] * 4}).to_csv(d / "sim_power_reps.csv", index=False)


def test_end_to_end_synthetic(tmp_path):
    daily = synthetic_daily(n_symbols=60, start="2018-01-01", end="2021-12-31", seed=2)
    daily.to_parquet(tmp_path / "daily.parquet", index=False)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["period"] = {"return_start": "2019-01", "return_end": "2021-12"}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    res = tmp_path / "results"
    _py("run.py", str(tmp_path / "config.yaml"), "--daily", str(tmp_path / "daily.parquet"),
        "--results-dir", str(res), "--cache-dir", str(tmp_path / "cache"), "--no-placebo", cwd=tmp_path)
    _toy_sims(tmp_path / "sims")
    _py("robustness_extra.py", str(tmp_path / "config.yaml"), "--results-dir", str(res),
        "--sims", str(tmp_path / "sims"), "--draws", "300", cwd=tmp_path)
    out = res / "upgrades" / "robustness"
    base = (res / "LATEST").read_text().strip()
    chk = pd.read_csv(out / "consistency_checks.csv")
    assert chk["match"].all(), chk
    boot = pd.read_csv(out / "bootstrap_ci.csv")
    assert set(boot["stat"]) >= {"hl_A", "hl_A_C", "hl_ResA", "hl_A_C_alpha", "fm1_lnA", "fm2_lnA_C",
                                 "fm3_ResLnA", "fm4_lnA_C", "fm4_ResLnA"}
    assert (boot["ci95_lo"] <= boot["ci95_hi"]).all() and (boot["n_draws"] == 300).all()
    loyo = pd.read_csv(out / "stability_loyo.csv")
    assert set(loyo["dropped_year"].astype(str)) == {"none", "2019", "2020", "2021"}
    man = json.loads((out / "run_manifest.json").read_text())
    assert man["base_run_id"] == base and man["seed"] == cfg["seed"] and man["consistency_all_match"]
    for k in ["git", "config_sha256", "code_sha256", "input_sha256", "timings_sec", "params"]:
        assert k in man
    led = pd.read_csv(out / "ledger.csv")
    assert {"base", "upgrade:robustness"} <= set(led["source"])
    assert "bootstrap_ci.csv" not in set(led["file"])
    summary = (out / "SUMMARY.md").read_text()
    assert base in summary and "Bootstrap" in summary
    # determinism: same seed -> same bootstrap table
    first = boot.copy()
    _py("robustness_extra.py", str(tmp_path / "config.yaml"), "--results-dir", str(res),
        "--sims", str(tmp_path / "sims"), "--draws", "300", cwd=tmp_path)
    pd.testing.assert_frame_equal(first, pd.read_csv(out / "bootstrap_ci.csv"))
    # --ledger-only reruns without touching the other tables
    _py("robustness_extra.py", str(tmp_path / "config.yaml"), "--results-dir", str(res), "--ledger-only", cwd=tmp_path)
    assert "ledger_rebuilt_utc" in json.loads((out / "run_manifest.json").read_text())
    summ = (out / "SUMMARY.md").read_text()                 # ledger section regenerated from the new ledger
    n_all = int(pd.read_csv(out / "ledger_summary.csv").set_index("source").loc["ALL", "n_unique"])
    assert f"unique tests {n_all};" in summ
    # if LATEST moves on, --ledger-only keeps using the manifest's base run (same as the other tables)
    led_before = pd.read_csv(out / "ledger.csv")
    (res / "LATEST").write_text("some_other_run\n")
    _py("robustness_extra.py", str(tmp_path / "config.yaml"), "--results-dir", str(res), "--ledger-only", cwd=tmp_path)
    pd.testing.assert_frame_equal(led_before, pd.read_csv(out / "ledger.csv"))
    assert base in (out / "SUMMARY.md").read_text()
