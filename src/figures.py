"""Key figures: quintile returns (paper vs us), H-L spreads with confidence intervals, t-statistics,
cumulative H-L returns, and the Amihud & Noh (2021) covariance term.

Usage:  python -I src/figures.py [--results-dir results] [--run-id RUN_ID (default: results/LATEST)]
Writes: results/<run_id>/fig_quintiles, fig_quintiles_demeaned, fig_hl_ci, fig_tstats, fig_hl_cumulative,
        fig_an_dif (if U1_* tables exist); each as .png and .pdf
All numbers come from the run's CSVs, src/paper_numbers.yaml and report.build().
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from report import PAPER, ROOT, T_CRIT, _hlz, build, resolve_run, sample_label

# validated categorical palette (dataviz reference instance, light mode), fixed order
C_PAPER, C_US, C_THIRD = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SIGNALS = [("A", "A", "A"), ("A_C", "AC", r"$A_C$"), ("ResA", "ResA", "Res. A")]
PAPER_LABEL = "Lou & Shu (2017), NYSE/AMEX 1964–2012"

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14, "xtick.labelsize": 13,
    "ytick.labelsize": 13, "legend.fontsize": 13, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "pdf.fonttype": 42,
})


def sample_note(rd: Path) -> str:
    """One-line description of our sample for figure footnotes (from sample.csv)."""
    s = rd / "sample.csv"
    if not s.exists():
        return "Ours: Binance spot USDT pairs."
    r = pd.read_csv(s).iloc[0]
    pm = yaml.safe_load(PAPER.read_text())["lou_shu_2017"]["sample"]["months"]
    return (f"Ours: Binance spot USDT pairs, returns {r['first']}–{r['last']} ({int(r['n_months'])} months, "
            f"{r['avg_coins_per_month']:.0f} coins/month on average). Paper: {PAPER_LABEL}, {pm} months. "
            "Signals measured in month t−2; equal-weighted.")


def quintile_data(rd: Path) -> tuple[dict, dict, dict]:
    """Raw EW quintile returns (%/mo) for the paper (Table 3A) and for us (T3_sorts), plus H-L (coef, t)."""
    t3 = yaml.safe_load(PAPER.read_text())["lou_shu_2017"]["table3_panelA"]
    paper = {s: np.array(t3[s]["raw"]["q"], float) for s, _, _ in SIGNALS}
    ours = pd.read_csv(rd / "T3_sorts.csv").set_index(["signal", "row"])
    us = {s: ours.loc[(s, "raw"), [f"Q{k}" for k in range(1, 6)]].to_numpy(float) for s, _, _ in SIGNALS}
    hl = {s: ((t3[s]["raw"]["hl"], t3[s]["raw"]["t"]),
              (float(ours.loc[(s, "raw"), "H-L"]), float(ours.loc[(s, "raw"), "t"]))) for s, _, _ in SIGNALS}
    return paper, us, hl


def _style(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if axis == "y":
        ax.axhline(0, color=INK2, linewidth=1.0)


def _footnote(fig, text: str):
    fig.text(0.01, 0.005, text, ha="left", va="bottom", fontsize=10.5, color=INK2, wrap=True)


def quintile_figure(paper: dict, us: dict, hl: dict, us_label: str, note: str, demean: bool, title: str,
                    ylabel: str):
    fig, axes = plt.subplots(1, 3, figsize=(13.33, 6.8), sharey=True)
    x = np.arange(1, 6)
    w = 0.38
    for ax, (s, _, name) in zip(axes, SIGNALS):
        p, u = paper[s], us[s]
        if demean:
            p, u = p - p.mean(), u - u.mean()
        ax.bar(x - w / 2 - 0.01, p, w, color=C_PAPER, label=PAPER_LABEL)
        ax.bar(x + w / 2 + 0.01, u, w, color=C_US, label=f"Ours, {us_label}")
        ax.set_title(f"Sorted on {name}")
        ax.set_xticks(x, ["1\nlow", "2", "3", "4", "5\nhigh"])
        ax.set_xlabel("Quintile of the signal")
        (pc, pt), (uc, ut) = hl[s]
        ax.text(0.02, 0.98, f"H–L, %/mo:\npaper {pc:.2f} (t {pt:.2f})\nours {uc:.2f} (t {ut:.2f})",
                transform=ax.transAxes, ha="left", va="top", fontsize=11.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=GRID))
        _style(ax)
    lo, hi = axes[0].get_ylim()
    axes[0].set_ylim(lo, hi + 0.35 * (hi - lo))          # room for the H-L box
    axes[0].set_ylabel(ylabel)
    h, l = axes[0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=17)
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.045))
    fig.tight_layout(rect=(0, 0.11, 1, 0.97))
    _footnote(fig, note)
    return fig


def hl_ci_figure(rd: Path, note: str):
    """H-L spread with 95% CI (±1.96·SE, SE = |H-L / t|): paper vs us, raw and alpha."""
    t3 = yaml.safe_load(PAPER.read_text())["lou_shu_2017"]["table3_panelA"]
    ours = pd.read_csv(rd / "T3_sorts.csv").set_index(["signal", "row"])
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 6.4), sharey=True)
    titles = {"raw": "Raw return", "alpha": "Alpha (paper: FF3+UMD; ours: CMKT+CMOM)"}
    y = np.arange(len(SIGNALS))[::-1]
    for ax, row in zip(axes, ["raw", "alpha"]):
        for k, (s, _, name) in enumerate(SIGNALS):
            for src, (c, t), col, dy in [("paper", (t3[s][row]["hl"], t3[s][row]["t"]), C_PAPER, 0.14),
                                         ("ours", (ours.loc[(s, row), "H-L"], ours.loc[(s, row), "t"]), C_US, -0.14)]:
                se = abs(c / t)
                ax.errorbar(c, y[k] + dy, xerr=T_CRIT * se, fmt="o", color=col, ms=9, elinewidth=2.2, capsize=4,
                            label=(PAPER_LABEL if src == "paper" else "Ours") if k == 0 else None)
                ax.annotate(f"{c:.2f} (t {t:.2f})", (c, y[k] + dy), xytext=(0, 9), textcoords="offset points",
                            ha="center", fontsize=10.5, color=INK)
        ax.axvline(0, color=INK2, linewidth=1.0)
        ax.set_ylim(y[-1] - 0.5, y[0] + 0.6)               # room for the value labels under the title
        ax.set_yticks(y, [n for _, _, n in SIGNALS])
        ax.set_title(titles[row], fontsize=14)
        ax.set_xlabel("Quintile 5 − quintile 1, % per month (95% CI)")
        _style(ax, axis="x")
    h, l = axes[0].get_legend_handles_labels()
    fig.suptitle("H–L spreads with 95% confidence intervals (Newey–West, 6 lags)", fontsize=17)
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.045))
    fig.tight_layout(rect=(0, 0.11, 1, 0.97))
    _footnote(fig, note)
    return fig


TSTAT_ROWS = [("HLArawCoef", "H–L on A (raw)"), ("HLACrawCoef", r"H–L on $A_C$ (raw)"),
              ("HLResArawCoef", "H–L on Res. A (raw)"), ("HLACalphaCoef", r"H–L on $A_C$ (alpha)"),
              ("FMlnA", "FM ln A (1)"), ("FMlnAC", r"FM ln $A_C$ (2)"), ("FMResLnA", "FM Res. ln A (3)"),
              ("FMlnACfour", r"FM ln $A_C$ (4)"), ("FMResLnAfour", "FM Res. ln A (4)"),
              ("TtwolnAbs", "FM ln|Ret| (T2, 3)"), ("TsixLnVOL", "FM ln DVol (T6)")]


def tstat_figure(df: pd.DataFrame, note: str):
    r = df.set_index("id")
    hlz = _hlz()
    rows = [(i, lab) for i, lab in TSTAT_ROWS if i in r.index]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(13.33, 7.0))
    for v in (-hlz, -T_CRIT, T_CRIT, hlz):
        ax.axvline(v, color=INK2, linewidth=1.2, linestyle="--" if abs(v) == hlz else ":")
    ax.axvline(0, color=INK2, linewidth=1.0)
    for k, (i, _) in enumerate(rows):
        pt, ut = r.loc[i, "paper_t"], r.loc[i, "us_t"]
        ax.plot([pt, ut], [y[k] + 0.12, y[k] - 0.12], color=GRID, linewidth=2, zorder=1)
        ax.scatter(pt, y[k] + 0.12, s=90, color=C_PAPER, zorder=3, label=PAPER_LABEL if k == 0 else None,
                   edgecolors=SURFACE, linewidths=2)
        ax.scatter(ut, y[k] - 0.12, s=90, color=C_US, zorder=3, label="Ours" if k == 0 else None,
                   edgecolors=SURFACE, linewidths=2)
    ax.set_yticks(y, [lab for _, lab in rows])
    ax.set_xlabel("Newey–West t-statistic, 6 lags (dotted: ±1.96; dashed: ±HLZ bar)")
    lim = max(6.0, float(np.nanmax(np.abs(r.loc[[i for i, _ in rows], ["paper_t", "us_t"]].to_numpy()))) + 0.5)
    ax.set_xlim(-lim, lim)
    ax.text(T_CRIT - 0.06, y[0] + 0.7, "±1.96", ha="right", fontsize=11, color=INK2)
    ax.text(hlz + 0.06, y[0] + 0.7, f"±{hlz:g} (HLZ)", ha="left", fontsize=11, color=INK2)
    ax.set_ylim(y[-1] - 0.7, y[0] + 1.2)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("t-statistics: paper vs ours", loc="left", fontsize=17)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _footnote(fig, note + " FM: paper FF3-adjusted, ours CMKT-adjusted returns.")
    return fig


def hl_cumulative(rd: Path, us_label: str):
    pr = pd.read_csv(rd / "T3_portfolios.csv")
    pr["date"] = pd.PeriodIndex(pr["month"], freq="M").to_timestamp()
    fig, ax = plt.subplots(figsize=(13.33, 6.2))
    colors = [C_PAPER, C_US, C_THIRD]
    for (s, _, name), c in zip(SIGNALS, colors):
        g = pr[pr["signal"] == s].sort_values("date")
        cum = g["HL"].fillna(0).cumsum()
        ax.plot(g["date"], cum, color=c, linewidth=2.2, label=f"H–L on {name}")
        ax.annotate(name, (g["date"].iloc[-1], cum.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=13)
    # mark the most influential month of the A_C spread
    g = pr[pr["signal"] == "A_C"].set_index("month")["HL"].dropna()
    top = (g - g.mean()).abs().idxmax()
    gg = pr[pr["signal"] == "A_C"].sort_values("date")
    cum = gg["HL"].fillna(0).cumsum().to_numpy()
    k = int(np.where(gg["month"].to_numpy() == top)[0][0])
    ax.annotate(f"{top}: H–L on $A_C$ {g[top]:+.0f} pp in one month\n"
                f"(mean over the other months {g.drop(top).mean():.2f}%/mo)",
                (gg["date"].iloc[k], cum[k]), xytext=(0.36, 0.49), textcoords="axes fraction", fontsize=12,
                color=INK, arrowprops=dict(arrowstyle="->", color=INK2),
                bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=GRID))
    _style(ax)
    ax.set_ylabel("Cumulative sum of monthly H–L, pp")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(f"Quintile 5 − quintile 1, EW raw returns, signal in t−2, {us_label}", loc="left", fontsize=17)
    ax.margins(x=0.06)
    fig.tight_layout()
    return fig


def an_figure(rd: Path, note: str):
    """Amihud & Noh (2021) term: DIF quintile returns (ours) and t-stats of the A&N regressions."""
    an = yaml.safe_load(PAPER.read_text())["amihud_noh_2021"]
    srt = pd.read_csv(rd / "U1_an_sorts.csv").set_index(["signal", "row"])
    fm = pd.read_csv(rd / "U1_an_fm.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.33, 6.4), gridspec_kw={"width_ratios": [1, 1.25]})
    q = srt.loc[("DIF", "raw"), [f"Q{k}" for k in range(1, 6)]].to_numpy(float)
    a1.bar(np.arange(1, 6), q, 0.6, color=C_US)
    a1.set_xticks(np.arange(1, 6), ["1\nlow", "2", "3", "4", "5\nhigh"])
    a1.set_xlabel("Quintile of DIF = ln A − ln|Ret| − ln $A_C$")
    a1.set_ylabel("Mean raw return, % per month")
    hl, t = srt.loc[("DIF", "raw"), "H-L"], srt.loc[("DIF", "raw"), "t"]
    a1.set_title(f"Ours: EW returns by DIF quintile\nH–L {hl:.2f}%/mo (t {t:.2f})", fontsize=14)
    _style(a1)

    def ours(model, var):
        r = fm[(fm["model"] == model) & (fm["var"] == var)]
        return float(r["t"].iloc[0]) if len(r) else np.nan
    rows = [("DIF (col. 3)", an["col3"]["DIF"]["t"], ours("AN: lnA_C+ln|Ret|+DIF", "DIF")),
            ("ln $A_C$ = ln IDVOL (col. 3)", an["col3"]["lnIDVOL"]["t"], ours("AN: lnA_C+ln|Ret|+DIF", "lnA_C")),
            ("ln|Ret| (col. 3)", an["col3"]["ln_absret"]["t"], ours("AN: lnA_C+ln|Ret|+DIF", "ln_absret")),
            ("Res. ln A = RlnILLIQ (col. 4)", an["col4"]["RlnILLIQ"]["t"], ours("AN: ResLnA+lnA_C+ivol", "ResLnA")),
            ("IVOL (col. 4)", an["col4"]["IdioVol"]["t"], ours("AN: ResLnA+lnA_C+ivol", "ivol"))]
    y = np.arange(len(rows))[::-1]
    hlz = _hlz()
    for v in (-hlz, -T_CRIT, T_CRIT, hlz):
        a2.axvline(v, color=INK2, linewidth=1.2, linestyle="--" if abs(v) == hlz else ":")
    a2.axvline(0, color=INK2, linewidth=1.0)
    for k, (lab, pt, ut) in enumerate(rows):
        a2.plot([pt, ut], [y[k] + 0.12, y[k] - 0.12], color=GRID, linewidth=2, zorder=1)
        a2.scatter(pt, y[k] + 0.12, s=90, color=C_PAPER, zorder=3, edgecolors=SURFACE, linewidths=2,
                   label="Amihud & Noh (2021), CRSP 1955–2016" if k == 0 else None)
        a2.scatter(ut, y[k] - 0.12, s=90, color=C_US, zorder=3, edgecolors=SURFACE, linewidths=2,
                   label="Ours" if k == 0 else None)
        a2.annotate(f"{pt:.2f} / {ut:.2f}", (1.0, y[k]), xycoords=("axes fraction", "data"), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=11, color=INK2)
    a2.text(1.0, y[0] + 0.55, "A&N / ours", transform=a2.get_yaxis_transform(), ha="left", fontsize=11,
            color=INK2)
    a2.set_ylim(y[-1] - 0.5, y[0] + 0.8)
    a2.set_yticks(y, [r[0] for r in rows])
    a2.set_xlabel("Newey–West t-statistic (dotted: ±1.96; dashed: ±HLZ bar)")
    a2.set_title("FM t-statistics, A&N Table 1 specifications", fontsize=14)
    a2.grid(axis="x", color=GRID, linewidth=0.6)
    h, l = a2.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.045))
    fig.suptitle("Amihud & Noh (2021): the omitted covariance term", fontsize=17)
    fig.tight_layout(rect=(0, 0.11, 0.97, 0.97))
    _footnote(fig, note + " A&N: 12-month measures, excess returns; ours: monthly, CMKT-adjusted.")
    return fig


def save(fig, rd: Path, name: str, out: list) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(rd / f"{name}.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    out.append(name)


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args(argv)
    rd = resolve_run(Path(a.results_dir), a.run_id)
    us_label = sample_label(rd)
    note = sample_note(rd)
    paper, us, hl = quintile_data(rd)
    made: list[str] = []
    save(quintile_figure(paper, us, hl, us_label, note, False,
                         "Equal-weighted monthly returns of quintile portfolios (Table 3, Panel A)",
                         "Mean raw return, % per month"), rd, "fig_quintiles", made)
    save(quintile_figure(paper, us, hl, us_label, note, True,
                         "Same, each source demeaned by its own average over the 5 quintiles",
                         "Return minus own cross-quintile mean, pp"), rd, "fig_quintiles_demeaned", made)
    save(hl_ci_figure(rd, note), rd, "fig_hl_ci", made)
    mf = rd / "run_manifest.json"
    man = json.loads(mf.read_text()) if mf.exists() else {}
    df, _, _ = build(rd, Path(a.results_dir) / "simulations", man)
    save(tstat_figure(df, note), rd, "fig_tstats", made)
    save(hl_cumulative(rd, us_label), rd, "fig_hl_cumulative", made)
    if (rd / "U1_an_sorts.csv").exists() and (rd / "U1_an_fm.csv").exists():
        save(an_figure(rd, note), rd, "fig_an_dif", made)
    print(f"figures written to {rd}: {', '.join(made)}")
    return rd


if __name__ == "__main__":
    main()
