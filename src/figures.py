"""Key figures: quintile returns (paper vs us) and cumulative H-L returns.

Usage:  python -I src/figures.py [--results-dir results] [--run-id RUN_ID (default: results/LATEST)]
Writes: results/<run_id>/fig_quintiles.(png|pdf), fig_quintiles_demeaned.(png|pdf), fig_hl_cumulative.(png|pdf)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from report import PAPER, ROOT, resolve_run, sample_label

# validated categorical palette (dataviz reference instance, light mode), fixed order
C_PAPER, C_US, C_THIRD = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SIGNALS = [("A", "A", "A"), ("A_C", "AC", r"$A_C$"), ("ResA", "ResA", "Res. A")]

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14, "xtick.labelsize": 13,
    "ytick.labelsize": 13, "legend.fontsize": 13, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "pdf.fonttype": 42,
})


def quintile_data(rd: Path) -> tuple[dict, dict]:
    """{signal: array of 5 raw EW returns, %/mo} for the paper (Table 3A) and for us (T3_sorts)."""
    t3 = yaml.safe_load(PAPER.read_text())["lou_shu_2017"]["table3_panelA"]
    paper = {s: np.array(t3[s]["raw"]["q"], float) for s, _, _ in SIGNALS}
    ours = pd.read_csv(rd / "T3_sorts.csv").set_index(["signal", "row"])
    us = {s: ours.loc[(s, "raw"), [f"Q{k}" for k in range(1, 6)]].to_numpy(float) for s, _, _ in SIGNALS}
    return paper, us


def _style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(0, color=INK2, linewidth=1.0)


def quintile_figure(paper: dict, us: dict, us_label: str, demean: bool, title: str, ylabel: str):
    fig, axes = plt.subplots(1, 3, figsize=(13.33, 6.2), sharey=True, layout="constrained")
    x = np.arange(1, 6)
    w = 0.38
    for ax, (s, _, name) in zip(axes, SIGNALS):
        p, u = paper[s], us[s]
        if demean:
            p, u = p - p.mean(), u - u.mean()
        ax.bar(x - w / 2 - 0.01, p, w, color=C_PAPER, label="Lou & Shu (2017), NYSE/AMEX 1964–2012")
        ax.bar(x + w / 2 + 0.01, u, w, color=C_US, label=f"Ours, {us_label}")
        ax.set_title(f"Sorted on {name}")
        ax.set_xticks(x, ["1\nlow", "2", "3", "4", "5\nhigh"])
        ax.set_xlabel("Quintile (signal in month t−2)")
        _style(ax)
    axes[0].set_ylabel(ylabel)
    h, l = axes[0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=18)
    fig.legend(h, l, loc="outside lower center", ncol=2, frameon=False)
    return fig


def hl_cumulative(rd: Path, us_label: str):
    pr = pd.read_csv(rd / "T3_portfolios.csv")
    pr["month"] = pd.PeriodIndex(pr["month"], freq="M").to_timestamp()
    fig, ax = plt.subplots(figsize=(13.33, 6.2))
    colors = [C_PAPER, C_US, C_THIRD]
    for (s, _, name), c in zip(SIGNALS, colors):
        g = pr[pr["signal"] == s].sort_values("month")
        cum = g["HL"].fillna(0).cumsum()
        ax.plot(g["month"], cum, color=c, linewidth=2.2, label=f"H–L on {name}")
        ax.annotate(name, (g["month"].iloc[-1], cum.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=13)
    _style(ax)
    ax.set_ylabel("Cumulative sum of monthly H–L, %")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(f"Quintile 5 − quintile 1, EW raw returns, {us_label}", loc="left", fontsize=18)
    ax.margins(x=0.06)
    fig.tight_layout()
    return fig


def save(fig, rd: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(rd / f"{name}.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args(argv)
    rd = resolve_run(Path(a.results_dir), a.run_id)
    us_label = sample_label(rd)
    paper, us = quintile_data(rd)
    save(quintile_figure(paper, us, us_label, False,
                         "Equal-weighted monthly returns of quintile portfolios (Table 3, Panel A)",
                         "Mean raw return, % per month"), rd, "fig_quintiles")
    save(quintile_figure(paper, us, us_label, True,
                         "Same, each source demeaned by its own average over the 5 quintiles",
                         "Return minus own cross-quintile mean, pp"), rd, "fig_quintiles_demeaned")
    save(hl_cumulative(rd, us_label), rd, "fig_hl_cumulative")
    print(f"figures written to {rd}")
    return rd


if __name__ == "__main__":
    main()
