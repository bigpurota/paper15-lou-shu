# Lou & Shu (2017) vs. us — run `20261016T134056Z_18099336`

Data: `/Users/purota/rs3-paper15/impl/data/cache/daily.parquet` (sha256 `216aac33fb7e`), git `8b603803ec` (dirty). Paper: NYSE/AMEX 1964–2012, 588 months; A&N (2021): CRSP 1955–2016.

| Metric | Paper | Us (Binance 2019-01–2026-09, 93 mo.) | Match? | Why it differs |
|---|---|---|---|---|
| corr(A, A_C) | 0.899 | 0.956 | yes | Same construction; level depends on the cross-sectional dispersion of \|r\| vs. DVol (D1). TBD after real data. |
| corr(A, \|Ret\|) | 0.489 | -0.133 | no | Level and dispersion of crypto \|r\| differ from NYSE/AMEX stocks (D1). TBD after real data. |
| corr(A_C, \|Ret\|) | 0.394 | -0.248 | no | As above (D1). TBD after real data. |
| Q75/Q25 of A_C | 116.60 | 9.398 | sign only | Dispersion of dollar volume across coins vs. stocks; the $10k volume filter cuts the illiquid tail (D8). TBD after real data. |
| Q75/Q25 of \|Ret\| | 2.182 | 1.746 | yes | Crypto volatility level and dispersion differ from stocks (D1). TBD after real data. |
| H-L on A, raw (%/mo) | 0.560 (t 2.36) | 3.286 (t 1.14) | sign only | Far fewer months than 588 and fewer assets -> lower power (D1); crypto returns are much more volatile than stock returns. TBD after real data. |
| H-L on A, alpha (%/mo) | 0.350 (t 2.31) | 2.261 (t 1.38) | sign only | Alpha on CMKT+CMOM instead of FF3+UMD (D6); short sample (D1). TBD after real data. |
| H-L on A_C, raw (%/mo) | 0.610 (t 2.95) | 57.52 (t 1.07) | sign only | Short sample, fewer assets (D1). TBD after real data. |
| H-L on A_C, alpha (%/mo) | 0.440 (t 3.20) | 2.583 (t 1.51) | sign only | CMKT+CMOM alpha (D6); short sample (D1). TBD after real data. |
| H-L on Res.A, raw (%/mo) | -0.170 (t -1.05) | -52.84 (t -1.03) | yes | Paper's spread is insignificant; ours tests the same null on a short sample (D1). TBD after real data. |
| H-L on Res.A, alpha (%/mo) | -0.160 (t -0.96) | 0.473 (t 0.26) | yes | CMKT+CMOM alpha (D6). TBD after real data. |
| FM ln A, model (1) | 0.119 (t 3.01) | 11.48 (t 1.08) | sign only | CMKT-adjusted returns, 36m betas (D4); no ln(ME), B/M controls (D2, D3); r_f = 0 (D5, no effect on slopes). TBD after real data. |
| FM ln A_C, model (2) | 0.183 (t 4.79) | 15.43 (t 1.08) | sign only | As for ln A (D2-D4). TBD after real data. |
| FM Res.lnA, model (3) | -0.383 (t -5.07) | -128.81 (t -1.06) | sign only | As for ln A (D2-D4). TBD after real data. |
| FM Res.lnA, model (5) with ln A_C + IVOL | -0.303 (t -4.65) | -169.01 (t -1.06) | sign only | IVOL vs daily EW CMKT, window ends in t-2 (D7); no size control (D2). TBD after real data. |
| A&N (2021) col. 4: RlnILLIQ with ln A_C + IVOL | 0.736 (t 4.63) | -169.01 (t -1.06) | no | Same regression as the row above; Amihud & Noh use 12-month measures and get the opposite sign to Lou & Shu. TBD after real data. |
| A&N (2021) col. 3: DIF with ln A_C + ln\|Ret\| | 0.996 (t 3.75) | -105.51 (t -0.92) | no | Our measures are monthly (A&N: 12-month windows), no size/B/M controls (D2, D3). A&N col. 2 (with ln LSIlliq) gives 1.219 (t 4.42). TBD after real data. |
| alpha of H-L(A) on IMLA_C (%/mo) | -0.040 (t -0.77) | 3.192 (t 1.12) | yes | IMLA_C built from our tercile sorts on A_C; one-factor alpha. TBD after real data. |

**Match rule.** Coefficients / spreads: if the paper is significant (|t| ≥ 1.96), *yes* = same sign and |t| ≥ 1.96, *sign only* = same sign but insignificant, *no* = opposite sign; if the paper is insignificant, *yes* = we are insignificant too, *sign only* / *no* = we are significant with the same / opposite sign. Levels (correlations, Q75/Q25): *yes* = same sign and within half an order of magnitude (|log10(us/paper)| ≤ 0.5), *sign only* = same sign otherwise. Deviations D1–D11: see `DEVIATIONS.md`.
