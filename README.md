# Lou & Shu (2017): Price Impact or Trading Volume?

**Team 12 · Paper 15 · HSE DSBA-3**

Replication and extension of Lou & Shu (2017), *Price Impact or Trading Volume: Why Is the Amihud (2002) Measure Priced?*

## Overview

This project investigates why illiquid assets may earn higher expected returns.

The original paper argues that the Amihud illiquidity premium is primarily explained by **low trading volume**, rather than the price impact of trades.

We test whether this conclusion holds in cryptocurrency markets, Russian equities, and more recent US stock market data.

## Methodology

We decompose the Amihud illiquidity measure into its components:

\[
A=\operatorname{mean}_d\left(\frac{|r_d|}{DVol_d}\right)
\]

\[
A_C=\operatorname{mean}_d\left(\frac{1}{DVol_d}\right)
\]

Here, \(A\) measures illiquidity, while \(A_C\) isolates inverse dollar trading volume.

Our analysis includes:

- Quintile portfolio sorts and long-short returns.
- Fama–MacBeth cross-sectional regressions.
- Newey–West standard errors.
- Direct price-impact estimation using high-frequency data.
- Simulations, robustness checks, and statistical power analysis.

## Data

| Market | Data source | Purpose |
|---|---|---|
| Crypto | Binance Spot | Main replication |
| Russia | MOEX ISS | Cross-market validation |
| US | Open Source Asset Pricing | Post-2012 comparison |

The main Binance dataset covers **626 cryptocurrencies**, **93 months** (January 2019 to September 2026), and **24,628 coin-month observations**.

Additional experiments use Binance 5-minute bars, aggregate trades, and market sentiment indicators.

## Key Findings

**1. Trading volume premium**

The original paper's positive volume-premium estimates do not replicate with conventional statistical significance in our full Binance sample.

The \(A_C\) quintile spread is 1.70% per month (t = 1.12), compared with 0.61% (t = 2.95) in the paper.

**2. Price impact versus volatility**

The residual component of Amihud illiquidity sometimes appears significantly priced. However, its significance disappears after controlling for absolute returns, suggesting that volatility may explain much of the relationship.

**3. Cross-market evidence**

MOEX provides partial support for the trading-volume hypothesis. US data suggest that the raw illiquidity premium weakened after 2012.

Direct price-impact estimates on Binance yield results that differ from the original study.

**4. Statistical reliability**

The tests have limited power to detect effects as small as those reported in the paper. Consequently, insignificant results cannot establish that a liquidity premium does not exist.

## Getting Started

Requires Python 3.12 and internet access for initial data downloads.

```bash
git clone https://github.com/bigpurota/paper15-lou-shu.git
cd paper15-lou-shu

make venv       # Install dependencies
make all        # Run the complete pipeline
make verify     # Verify reproducibility
make slides     # Generate presentation
```

The high-frequency upgrade downloads several gigabytes of market data. Building the slides also requires TeX Live and Arial.

## Repository Structure

```text
src/            Data processing and analysis
tests/          Unit and integration tests
results/        Tables, figures and experiments
slides/         Presentation materials
config.yaml     Experiment configuration
DEVIATIONS.md   Methodological differences
Makefile        Reproducible commands
```

## Limitations

The Binance sample is shorter and more volatile than the original equity sample. Cryptocurrency markets differ structurally from equities, and market-capitalization controls are unavailable for the main sample.

Some results are sensitive to individual periods, particularly March 2021. Several additional specifications were introduced after observing initial results.

## Conclusion

Our results **do not conclusively confirm or reject** Lou & Shu's trading-volume explanation.

The findings highlight the difficulty of separating trading volume, volatility, and price impact, particularly across different asset classes.

For detailed results, see `results/`. For methodological differences and research decisions, see `DEVIATIONS.md`.

## Team

Andrey Tsember · Sofia Shapovalova · Ksenia Stepanova · Marat Bineman · Alexia Fox

**References:** Lou & Shu (2017); Amihud & Noh (2021).
