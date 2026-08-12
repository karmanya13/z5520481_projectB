# AGENTS.md — FINS3645 Part B Agent Instructions

## Project

This repository is my UNSW FINS3645 FinTech Project Part B for z5520481.

Product: **No Bull Investing**

The project contains:
- 12 systematic funds across Equity, Crypto and Combined families
- Equal Weight, Minimum Variance, Maximum Sharpe and Risk Parity methods
- walk-forward out-of-sample backtests
- a Finance-Adjusted VADER sentiment extension
- a lagged sentiment-fusion test
- a Streamlit investor interface

Read `PROJECT_BRIEF.md` and inspect the current project files before changing anything.

Do not restart completed work or replace locked methodology unless a genuine methodological error is identified.

---

## Locked Portfolio Rules

Keep these decisions unless an error is demonstrated:

- monthly rebalancing
- first eligible trading day of each month
- 252-observation rolling estimation window
- no look-ahead
- estimation data strictly earlier than the rebalance date
- long-only weights
- weights sum to 1
- risk-free rate = 0
- transaction costs = 0 in the current model
- natural holdings drift between monthly rebalances
- Equity annualisation = 252
- Combined annualisation = 252
- Crypto annualisation = 365
- Combined funds use the equity trading calendar
- weekend-only crypto returns are not included in the Combined return series

Never reapply monthly target weights every day.

---

## Sentiment Rules

The sentiment baseline is Standard VADER.

The main innovation is a 32-term Finance-Adjusted VADER lexicon.

Keep these rules:

- preserve raw headline text
- do not overwrite existing VADER terms
- ticker-days with no news receive sentiment = 0
- interpret no-news zero as no new measured news signal
- sector sentiment is equal-weighted across the five tickers in each sector
- trading uses one-trading-day-lagged sentiment
- never use same-day sentiment for a trading decision
- sentiment tilt strength is fixed at `1.0`
- do not tune the tilt after seeing OOS results to manufacture outperformance

Fusion uses Equity Equal Weight as the baseline.

---

## Coding Workflow

For any Python change:

1. inspect the current file first,
2. make the smallest auditable change possible,
3. compile the file,
4. run the relevant script,
5. inspect printed output,
6. verify generated CSVs/figures,
7. only then continue.

Do not trust plausible output simply because the code runs.

Keep reusable logic in `src/` and build/validation work in `scripts/`.

Do not commit raw course data.

The Streamlit app should read precomputed artifacts and must not rerun the full backtest or VADER pipeline on page load.

---

## Validation Rules

Check:

- sample dates,
- annualisation conventions,
- portfolio-weight drift,
- concentration,
- CAGR versus Sharpe definitions,
- sentiment lagging,
- percentage versus percentage-point wording,
- figure/table consistency,
- report claims against actual outputs.

Known project lesson:
an earlier portfolio implementation accidentally reapplied target weights every day. The outputs looked plausible, but auditing revealed the mismatch. The corrected implementation uses natural holdings drift between monthly rebalances.

---

## Reporting Rules

Do not invent:
- results,
- headline examples,
- predictive claims,
- transaction-cost results,
- turnover results,
- live-trading claims.

Distinguish evidence from inference.

Report negative findings honestly.

The Finance-Adjusted VADER result should be framed as a targeted finance-language extension, not as proof of improved sentiment accuracy without labelled ground truth.

AI may assist with coding, checking and drafting support, but final submitted interpretation and assessment decisions must be reviewed by me.

---

## AI Workflow Evidence

Important AI-assisted episodes are documented under `ai/`.

Each episode should show:
- what was asked,
- what AI suggested,
- what I checked,
- what was wrong or incomplete,
- what I changed,
- why,
- final outcome.

Do not conceal AI use and do not treat AI output as automatically correct.
