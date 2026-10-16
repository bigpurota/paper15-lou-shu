# Upgrade 3: the EDGE spread and the economic significance of the premium

Base run: `20261016T221825Z_3e8eb87e`; upgrade run: `20261016T221940Z_3e8eb87e_edge_spread`. Every number below comes from the CSVs in this folder.

## What we tested
- EDGE (Ardia, Guidotti, Kroencke 2024): our own implementation on daily OHLC within the month (≥10 pairs of adjacent days). Cross-check against the `bidask` package 2.1.0: 300 coin-months (random draw stratified so that up to a third are months with missing days; 6 of them have gaps; `edge_bidask_check.csv`), max |difference| = 0.0e+00.
- Non-positive estimates: 50.7% of coin-months with an estimate (median EDGE -0.033%): on daily candles a coin's monthly spread estimate is almost pure noise. The main rule is floor (the smallest positive estimate of the month); robustness: drop, abs and a 12-month window (all variants in the CSVs).
- Analogue of Table 9 Panel C (EDGE instead of ES), analogue of Table 10 (cost/non-cost parts of ln A), long–short costs of the A_C quintiles.

## Key numbers (floor rule) vs the paper
| Statistic | Paper (ES, NYSE/AMEX) | Us (EDGE, Binance) |
|---|---|---|
| corr(spread, A) | 0.731 | 0.181 |
| corr(spread, A_C) | 0.705 | 0.158 |
| corr(spread, Res.A) | 0.215 | 0.109 |
| FM: ln spread alone | -0.248 (t -1.94) | -0.017 (t -0.04) |
| FM: ln A_C together with the spread (paper: model 4, + QS, ES, RS, λ) | 0.210 (t 3.71) | 0.762 (t 1.51); ln EDGE -0.224 (t -0.72) |
| T10: cost alone | -0.047 (t -1.11) | -10.108 (t -1.07) |
| T10: non-cost alone | 0.186 (t 3.04) | 0.547 (t 1.17) |
| T10: together, cost / non-cost | 0.015 (t 0.35) / 0.191 (t 3.10) | -9.675 (t -1.05) / 0.570 (t 1.22) |
| 12-month window: corr(EDGE, A_C); ln EDGE in FM | — | 0.188; 0.519 (t 1.94) |

## Premium vs costs (A_C quintiles, EW, %/month; leg half-spread = sqrt(mean s²)/2)
- Gross H–L: 1.65 (t 1.07); turnover long/short: 0.62/0.46 per month; mean half-spread long/short: 0.11%/0.03%.
- Long–short costs: 0.08%/month; net H–L: 1.57 (t 1.02), 18.8%/year; cost cushion: ×20.51.
- Conservative variant (|EDGE| per coin, abs rule; noise inflates it relative to EDGE, but it is not a guaranteed upper bound on the true spread): 0.22%/month, net H–L 1.43 (t 0.93).
- Reliability of the main estimate: in 42% of months for the long leg and 54% for the short leg the leg's mean s² is ≤ 0 and its half-spread is set to 0, so daily-candle EDGE cannot resolve the spread even at the portfolio level; the cost figure must not be read as a measurement.
- EDGE-free bound: the cost per side (half-spread + fee) at which the mean net H–L is 0: 1.52% (at a long+short turnover of 1.08 per month).
- Extra variant (not pre-registered): + a fee of 0.10% per side = 0.11%/month; net H–L after spread and fee 1.46 (t 0.95), alpha 0.54 (t 0.43).

## Interpretation
- ln EDGE alone: not significant (|t| < 1.96 < 3 HLZ) (sign −); ln A_C controlling for EDGE: not significant (|t| < 1.96 < 3 HLZ).
- Non-cost part of ln A: not significant (|t| < 1.96 < 3 HLZ); cost part: not significant (|t| < 1.96 < 3 HLZ). The paper's pattern (only the non-cost part is priced) is not statistically reproduced on our data.
- Across all 6 variants (window × rule): ln EDGE alone > 0 in 5 of 6 (the paper's sign is −); cost part > 0 in 5 of 6; non-cost > 0 in 6 of 6. Max |t| over EDGE / A_C / cost / non-cost across all variants = 1.94 (< 1.96). The paper's sign pattern (cost ≤ 0, non-cost > 0) holds in 1 of 6 variants.
- The premium is positive after costs but not statistically significant (nor is the gross one).

## Caveats
- EDGE is an estimate from daily candles, not an observed spread; for liquid coins the true spread is small and the estimate is noisy (hence negative s²). The task (task.md, line 59) envisages hourly bars; this is a deviation.
- Crypto trades 24/7: the median |open − previous day's close| = 3.3 bp (open = previous close on 35% of days) against a median daily high−low range of 723 bp. The median weight of EDGE's second moment (on this gap) = 0.9996: the estimate is driven almost entirely by the bounce between the last trade of one day and the first trade of the next.
- T10, cost part: Cost = a_t + b_t·ln EDGE, so the monthly coefficient = (coefficient on ln EDGE)/b_t. The slope b_t < 0 in 10% of months, min |b_t| = 0.0005; monthly cost coefficients: median -0.96, max |·| 831. The FM mean of the cost part is driven by a few months, so only t, not magnitude, can be compared with the paper.
- Costs: the main estimate is the half-spread only (the fee is a separate line above); it excludes the price impact of large orders and the cost/availability of shorting (many illiquid coins cannot be borrowed).
- The decomposition of ln A into cost/non-cost is estimated on the panel of coins with a return in t (as in FM), not on the full t−2 cross-section.
