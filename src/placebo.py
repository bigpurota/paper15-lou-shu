"""Placebo tests (task.md 13.1): the pipeline must lose the premium when signals are randomly
re-assigned, and must detect a deliberate look-ahead leak.

(a) shuffle: within each month t, the block of t-2 signals [A, A_C, ResA, lnA, lnA_C, ResLnA, ivol]
    is permuted across coins as ONE block (a coin receives all signals of one other coin, so the
    cross-signal structure, e.g. ResLnA orthogonal to lnA_C, is kept); returns and controls
    (ret_adj, ret, mom, ret_1) stay with their own coin. FM models (2), (3), (4) of Table 4 on
    ret_adj and raw quintile H-L sorts on A, A_C, ResA are re-run on each of `n_perm`
    permutations (seeded). Expectation: mean coef ~ 0, share |t| > 1.96 ~ 5%.
(b) leak (only if `panel_lag0` is given): the same tests on a panel whose signals are measured in
    month t itself (build_panel with measures.lag_months = 0, see `lag_panel`), i.e. with
    contemporaneous information. Expectation: a clearly different, typically much stronger
    relation than the lag-2 baseline; on real data |r| and volume of month t co-move with the
    return of month t. (On simulated data, where returns do not feed back into volume, the leak
    panel only differs by persistence, so a strong effect is not expected there.)
Rows with test == "baseline" repeat the same statistics on the lag-2 panel for comparison.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from analysis import build_panel
from fama_macbeth import fama_macbeth
from sorts import sort_table

SIGNALS = ["A", "A_C", "ResA", "lnA", "lnA_C", "ResLnA", "ivol"]
FM_MODELS = {"(2)": ["lnA_C"], "(3)": ["ResLnA"], "(4)": ["lnA_C", "ResLnA"]}
SORTS = ["A", "A_C", "ResA"]


def lag_panel(daily: pd.DataFrame, cfg: dict, start: str, end: str, lag: int) -> pd.DataFrame:
    """Panel with signals of month t-lag (lag=0: contemporaneous leak, lag=-1: future)."""
    c = copy.deepcopy(cfg)
    c["measures"]["lag_months"] = int(lag)
    return build_panel(daily, c, start, end)[0]


def _stats(panel: pd.DataFrame, cfg: dict) -> list[dict]:
    lags, q = cfg["tests"]["nw_lags"], cfg["tests"]["quantiles"]
    controls = ["mom", "ret_1"]
    rows = []
    for model, xs in FM_MODELS.items():
        res = fama_macbeth(panel, "ret_adj", xs + controls, lags=lags).set_index("var")
        for x in xs:
            rows.append({"var": f"FM{model} {x}", "coef": res.loc[x, "coef"], "t": res.loc[x, "t"]})
    for s in SORTS:
        tab = sort_table(panel, s, "ret", q, None, lags)
        rows.append({"var": f"H-L {s}", "coef": float(tab.loc["raw", "H-L"]), "t": float(tab.loc["raw", "t"])})
    return rows


def shuffle_signals(panel: pd.DataFrame, rng: np.random.Generator, by: str = "month") -> pd.DataFrame:
    """Permute the signal block across rows within each cross-section."""
    out = panel.copy()
    cols = [c for c in SIGNALS if c in out]
    vals = out[cols].to_numpy()
    new = np.empty_like(vals)
    for idx in out.groupby(by).indices.values():
        new[idx] = vals[rng.permutation(idx)]
    out[cols] = new
    return out


def run_placebos(panel: pd.DataFrame, factors: pd.DataFrame, cfg: dict, seed: int,
                 panel_lag0: pd.DataFrame | None = None, n_perm: int = 20) -> pd.DataFrame:
    """Columns: test, var, coef, t, expectation (+ n, sd_t, share_abs_t_gt_1.96 for the shuffle).

    `factors` is accepted for interface symmetry with run_tests; placebo sorts use raw returns.
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 777]))
    rows = [{"test": "baseline (lag 2)", **r, "expectation": "reference", "n": 1} for r in _stats(panel, cfg)]

    perms = pd.DataFrame([{**r, "perm": i} for i in range(n_perm)
                          for r in _stats(shuffle_signals(panel, rng), cfg)])
    for var, g in perms.groupby("var", sort=False):
        rows.append({"test": "shuffle within month", "var": var, "coef": g["coef"].mean(),
                     "t": g["t"].mean(), "expectation": "coef ~ 0, share |t|>1.96 ~ 5%", "n": len(g),
                     "sd_t": g["t"].std(ddof=1), "share_abs_t_gt_1.96": float((g["t"].abs() > 1.96).mean())})

    if panel_lag0 is not None:
        rows += [{"test": "leak (lag 0)", **r, "n": 1,
                  "expectation": "differs from baseline; strong |t| on real data = leak detectable"}
                 for r in _stats(panel_lag0, cfg)]
    return pd.DataFrame(rows)[["test", "var", "coef", "t", "expectation", "n", "sd_t", "share_abs_t_gt_1.96"]]
