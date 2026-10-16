# Speaker roles and Q&A ownership: Team 12, Paper 15 (Lou & Shu 2017)

Defence: Saturday 17 October 2026, 13:00, room G116. Talk: 15 minutes, then questions.
Deck: `slides/deck.tex` → `slides/paper15_team12_LouShu2017.pdf` (the footer of every main slide names its speaker; every backup slide names its Q&A owner). Speaker notes per slide: `slides/PLAN.md`.

Team (from `slides/meta.yaml`; check the Latin spelling with each member): Andrey Tsember, Sofya Shapovalova, Ksenia Stepanova, Marat Bineman, Alexia Fox.

This file contains no results. For any number, use the slide or PLAN.md, where every number is generated from `results/` by `src/build_docs.py`.

## 1. Speaking order and timing

The course template is Problem 2 / Method 3 / Results 5 / Critique 3 / Next steps 2 minutes. We follow it within ±30 s: the title and literature add 30 s to Problem, and Critique gets 30 s less because the overfitting slide sits in the last block.

| # | Speaker | Slides (deck = PLAN numbering) | Template block | Planned time | Target (with buffer) |
|---|---|---|---|---|---|
| 1 | Marat Bineman | 1 Title · 2 Problem · 3 Key observation (A vs A_C, dispersion of volume vs \|r\|, corr(A, A_C)); literature line on slide 2 | Problem | 2:30 (0:15 + 1:00 + 1:15) | 2:15 |
| 2 | Sofya Shapovalova | 4 Method: A_C and the residual, simulations V/P/0 · 5 Tests and data (sorts, Fama–MacBeth, NW(6), Binance universe, survivorship, redenomination audit, deviations) | Method | 3:00 (1:30 + 1:30) | 2:45 |
| 3 | Ksenia Stepanova | 6 The paper's results · 7 Our replication on the same axes · 8 Paper versus us, why results differ, power/MDE | Results | 3:00 (0:45 + 1:15 + 1:00) | 3:00 |
| 4 | Andrey Tsember | 9 HF λ with taker-volume signs (EDGE, MOEX, OpenAP in backup B5) · 10 Sentiment, market illiquidity, January · 11 Amihud & Noh: who is right | Results (9–10) + Critique (11) | 3:00 (1:15 + 0:45 + 1:00) | 3:00 |
| 5 | Alexia Fox | 12 Critique · 13 Overfitting (required slide) · 14 Next steps | Critique (12) + Overfitting + Next steps | 3:30 (1:30 + 1:15 + 0:45) | 3:30 |
| | | | | **15:00** | **14:30** |

The planned times sum to exactly 15:00 and leave no buffer, so Marat and Sofya aim 15 s under their planned time. **A timed rehearsal has not been done yet.** After the rehearsal, record the measured time per speaker in `../shared/STATUS.md` and adjust here.

If running over time, cut in this order: (1) slide 10 to its message line only (Andrey); (2) the bullets of slide 9 (Andrey); (3) the backup-only details on slide 13 (Alexia keeps the ledger, world-0 size and placebo lines). Never cut slide 7 (the required same-axes figure) or slide 13 (the required overfitting slide).

Hand-overs (one sentence each, so the story stays continuous):
- Marat → Sofya: "So A is almost only volume. Sofya will show how the paper separates volume from the rest."
- Sofya → Ksenia: "With the method checked on simulations, Ksenia compares the paper's results with ours."
- Ksenia → Andrey: "Our replication has the paper's signs but no power. Andrey shows what the upgrades add."
- Andrey → Alexia: "The data do not settle the A&N dispute. Alexia sums up the critique and the overfitting checks."

Marat opens the Q&A by naming who takes which topic (section 2) and passes each question to its owner. If the owner is stuck, the second owner answers.

## 2. Q&A ownership

| Member | Q&A topics (first owner) | Code they must be able to explain | Backup slides |
|---|---|---|---|
| Marat Bineman | What the paper claims (thesis in one sentence); place in the literature (Amihud 2002 → Goyenko–Holden–Trzcinka 2009 → Lou & Shu 2017 → Amihud & Noh 2021 → EDGE 2024); why lag t−2 (the paper only follows Brennan–Huh–Subrahmanyam 2013; our reason is bid–ask bounce and one-month reversal); why logs; why adjusted returns. PLAN questions 1–3. | No module of his own; must know the README run section and the paper's Tables 1, 3, 4 (`src/paper_numbers.yaml` holds the paper's numbers). | none (slides 1–3) |
| Sofya Shapovalova | Definitions of A, A_C, Res.A/Res.lnA; residual logic; Fama–MacBeth and sorts; why NW(6) and its size in world 0; the Binance universe and survivorship (later-delisted pairs kept, their count not recorded, archive completeness not checked, no delisting returns); the redenomination audit; DEVIATIONS (D, U, J entries); simulations V/P/0 and what failed; reproducibility (`make all`, manifests; the clean-clone check in `results/repro/` covers p0 and the light upgrades from an older commit, not hf_lambda, the ledger or the docs; a full `make all` from a clean clone is pending). PLAN questions 4, 13, 15. | `src/measures.py`, `src/analysis.py`, `src/data_binance.py`, `src/data_audit.py`, `src/simulate.py`, `src/fama_macbeth.py`. Added by us (not in the original brief): `src/run.py`, `config.yaml`, `Makefile` (the pipeline and pre-registration). | B1 Definitions, B2 Simulations, B6 Reproducibility |
| Ksenia Stepanova | The paper's quintiles against ours on the same axes; the paper-versus-us table and the match rule (yes / sign only / no); why the results differ (power, market, missing size and B/M); power and MDE80; EW vs VW; dependence on March 2021. PLAN questions 2 (second owner), 8, 9, 11. | `src/sorts.py`, `src/report.py`, `src/figures.py`, the power/MDE and bootstrap parts of `src/robustness_extra.py`. Added by us: `src/factors.py` (CMKT, CMOM, betas used in the alphas) and `src/build_docs.py` (how every slide number is generated). | B3 t-statistics, B4 Power and robustness |
| Andrey Tsember | HF λ with taker-volume signs and its validation against aggTrades; EDGE and costs; Amihud & Noh covariance term and "who is right"; MOEX; OpenAP US after 2012 (raw L−S vs FF3+UMD alpha); sentiment and market illiquidity; January. PLAN questions 5, 6, 7 (first owner), 10, 14 (first owner). | `src/hf_lambda.py`, `src/spread_edge.py`, `src/amihud_noh.py`, `src/run_moex.py` with `src/data_moex.py`, `src/openap_us.py`, `src/sentiment.py`. | B5 Other upgrades |
| Alexia Fox | Critique (assumptions, what the proxy error correlates with, missing baselines); overfitting: data snooping and multiple testing (the ledger, Bonferroni/BH, what the survivors really are), look-ahead checks (`tests/test_alignment.py`, lag-0 placebo), world-0 size, bootstrap, out-of-sample vs robustness; next steps. PLAN questions 7 (second owner), 11 (second owner), 12, 14 (second owner). | The ledger, bootstrap, stability and world-0 parts of `src/robustness_extra.py`; `src/placebo.py`; `tests/test_alignment.py`. | the overfitting material on slide 13; B7 What was not done and limitations |

## 3. Everyone must know (course criterion G: any member may be asked about any slide)

Every member must be able to give a short answer to each of these without passing the question on:

1. The thesis in one sentence: the authors claim the Amihud premium is a volume premium, not a price-impact premium, because A_C is priced and the residual Res.lnA is not positively priced; the weakest step is reading the residual as "price impact" when it is mostly |Ret|.
2. The definitions: A = mean(|r|/DVol), A_C = mean(1/DVol), Res.lnA = monthly residual of ln A on ln A_C; signal from month t−2, return in month t; NW(6) t-statistics; HLZ threshold t > 3.
3. What we replicated and what not: signs mostly match; our volume premium is insignificant; the significant rows are mainly Res.lnA, some of them only at 1.96 and not at the world-0 cut-off; nothing passes HLZ (slide 8).
4. Why results differ: far fewer months (power, MDE80), a different market (volume rises with volatility), no size or B/M (CMKT instead of FF3).
5. The A_C spread rests on March 2021; without it the spread is close to zero (slide 7).
6. Simulations: worlds V and P are told apart in FM; world 0 shows NW(6) over-rejects (a failed P0 criterion), which is why we also report the empirical cut-off.
7. One line per upgrade: λ has a negative FM slope in one 36-month window (NW size at that length not calibrated), unlike Table 9 model (1), and the A_C premium is absent there (ln A_C without λ is negative and insignificant), so it may be a window effect (only the signs in model (2) agree); the A&N DIF term is insignificant, and with coefficients on a different scale we cannot say whether we could detect A&N's effect; MOEX repeats the positive ln A_C slope at 1.96 but not the negative Res.lnA; OpenAP's raw Illiquidity fell after 2012, its FF3+UMD alpha is insignificant in both periods and the SMB loading fell (size exposure is a hypothesis, not tested); for sentiment the L−S spread is insignificant while the main FM slope passes 1.96 only marginally (not HLZ), so never say “no link”; market illiquidity has the paper's sign but is fragile. Post hoc diagnostics (λ checks, MOEX same-months check, without the most influential month) are always called post hoc.
8. Honesty limits: upgrade specifications and verdict rules were not pre-registered in git (only the base config.yaml was); the Res.lnA significance appears only after the J1 data audit, whose rules were designed after the first run; post hoc variants are labelled; a clean-clone check reproduced the base and light-upgrade CSVs from committed code, but the full `make all` from a clean clone (with hf_lambda, ledger and docs) is pending; delisting returns are missing; the paper's numbers were transcribed, not tested against the PDF; the slide deadline (24 h before the defence) was missed; the timing rehearsal has not been done.
9. Where things live: `README.md` (run, comparison table, deviations), `DEVIATIONS.md`, `results/<run_id>/paper_vs_us.md`, `results/upgrades/*/SUMMARY.md`, `src/` (all logic), `tests/`.

## 4. Before the defence

- [ ] Each member confirms the spelling of their name in `slides/meta.yaml`, then `make docs slides` is rerun.
- [ ] Timed rehearsal with a stopwatch; times per speaker written to `../shared/STATUS.md`.
- [ ] Each member reads the module docstrings of their code list above and runs `make tests` once.
- [ ] Each member answers the "Everyone must know" list aloud to another member.
