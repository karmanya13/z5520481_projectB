# No Bull Investing — FINS3645 FinTech Project Part B

UNSW FINS3645: Financial Market Data Design & Analysis  
Student: Karmanya Singh (z5520481)

## Project Overview

No Bull Investing is a Streamlit-based prototype for comparing systematically managed multi-asset funds and exploring finance-specific news sentiment analytics.

The project implements twelve out-of-sample funds across three asset families:

- Equity
- Crypto
- Combined Equity + Crypto

Each family is evaluated using four portfolio-construction methods:

- Equal Weight
- Minimum Variance
- Maximum Sharpe
- Risk Parity

The project also extends Standard VADER with a 32-term Finance-Adjusted VADER lexicon and tests whether the resulting lagged sector sentiment signal improves an Equity Equal Weight portfolio.

## Main Features

The Streamlit app supports four parts of the investor journey:

1. **Compare Funds**
   - Compare CAGR, volatility, Sharpe ratio and maximum drawdown across all twelve funds.
   - View return-versus-risk results across Equity, Crypto and Combined strategies.

2. **Fund Fact Sheet**
   - Inspect realised out-of-sample growth of $1.
   - Review risk metrics and latest available target holdings.
   - Identify portfolio concentration.

3. **Build an Allocation**
   - Select up to four funds.
   - Assign starting portfolio percentages.
   - View blended historical performance over the common out-of-sample period.

4. **Sentiment Analytics**
   - Compare Standard VADER and Finance-Adjusted VADER.
   - Explore sector sentiment over time.
   - Review finance-term coverage, neutral-rate changes and news coverage.

## Methodology Summary

Portfolio backtests use:

- monthly rebalancing,
- first eligible trading day of each month,
- a 252-observation rolling estimation window,
- long-only weights,
- no look-ahead,
- zero risk-free rate,
- zero transaction costs in the current model,
- natural portfolio-weight drift between monthly rebalance dates.

Annualisation conventions:

- Equity: 252
- Combined: 252
- Crypto: 365

The Combined funds use the equity trading calendar, so weekend-only crypto movements are not included in the Combined return series.

## Sentiment Extension

The Part B innovation is a targeted Finance-Adjusted VADER extension.

The final lexicon contains 32 directional finance terms selected to improve sensitivity to financial headline language without replacing the broader Standard VADER model.

Key validated outputs include:

- 146,830 mapped headlines,
- Standard VADER neutral rate: 48.85%,
- Finance-Adjusted neutral rate: 45.96%,
- neutral-rate reduction: 2.88 percentage points,
- 4,257 Standard-neutral headlines rescued by the Finance-Adjusted model.

The fusion test uses a one-trading-day-lagged sector signal and a fixed `tilt_strength = 1.0`.

## Repository Structure

- `streamlit_app.py` — Streamlit app entry point
- `src/` — reusable portfolio, ETL, sentiment and fusion logic
- `scripts/` — build, validation and figure-generation scripts
- `results/data/` — precomputed data consumed by the app
- `results/tables/` — performance and validation tables
- `results/figures/` — report figures
- `report/report.docx` — editable report source
- `report/report.pdf` — final report
- `ai/` — curated AI workflow evidence
- `CLAUDE.md` / `AGENTS.md` — project-specific AI working instructions
- `context/` — supplied project/data context
- `tests/` — smoke tests

## Reproduce the Project

Install the application and development dependencies:

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

Rebuild the Part B outputs:

```bash
python scripts/run_part_b.py
```

Generate the report figures:

```bash
python scripts/make_figures.py
```

Run the Streamlit app locally:

```bash
python -m streamlit run streamlit_app.py
```

Run the hand-in checker:

```bash
python scripts/check_handin.py
```

## Deployment

Live app:

https://no-bull-investing-z5520481.streamlit.app/

Public GitHub repository:

https://github.com/karmanya13/z5520481_projectB

The deployed application reads the precomputed artifacts under `results/` and does not rerun the full backtest or VADER pipeline on page load.

## Data

Raw course data is not committed to this repository.

The project loads the hosted course data through `src/data_access.py`. Derived outputs required by the Streamlit application are committed under `results/`.

## Notes

This is an academic prototype and historical backtest. It does not execute live trades and does not guarantee future investment performance.
