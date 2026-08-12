# CLAUDE.md — FINS3645 Part B Working Instructions

## Project

This repository is my UNSW FINS3645 FinTech Project Part B for z5520481.

Product name: **No Bull Investing**

The project builds and evaluates systematic investment funds across:
- Equity
- Crypto
- Combined Equity + Crypto

Each family contains:
- Equal Weight
- Minimum Variance
- Maximum Sharpe
- Risk Parity

The Streamlit prototype allows a user to:
1. compare funds,
2. inspect a fund fact sheet,
3. build an allocation across funds,
4. inspect sentiment analytics.

Before making changes, read `PROJECT_BRIEF.md` and inspect the current project files. Do not recreate completed work.

---

## Source of Truth

Use the current project files and generated outputs as the source of truth.

Important locations:
- `src/` — reusable project logic
- `scripts/` — build, validation and figure scripts
- `results/data/` — precomputed app data
- `results/tables/` — performance and validation tables
- `results/figures/` — report figures
- `streamlit_app.py` — app entry point
- `report/` — report source/output
- `ai/` — AI workflow evidence

Never invent results, outputs, filenames, metrics or methodology.

If prompt text conflicts with current code or outputs, inspect the files and report the discrepancy before changing anything.

---

## Locked Portfolio Methodology

Do not change these decisions unless a genuine methodological error is demonstrated:

- 12 funds total:
  - Equity, Crypto and Combined
  - Equal Weight, Minimum Variance, Maximum Sharpe and Risk Parity
- Monthly rebalancing
- First eligible trading day of each month
- 252-observation rolling estimation window
- No look-ahead
- Estimation data must be strictly earlier than the rebalance date
- Long-only weights
- Weights sum to 1
- Risk-free rate = 0
- Transaction costs = 0 in the current model
- Natural holdings drift between monthly rebalances
- Target weights reset only at monthly rebalances
- Equity annualisation = 252
- Combined annualisation = 252
- Crypto annualisation = 365
- Crypto returns are calculated on the native crypto calendar before alignment
- Combined funds use the equity trading calendar, so weekend-only crypto returns are excluded from the Combined series

Never accidentally reapply monthly target weights every day.

---

## Portfolio Metrics

Use the existing definitions consistently:

- Annualised return = CAGR from realised growth of $1
- Annualised volatility = daily standard deviation × sqrt(annualisation factor)
- Sharpe = annualised arithmetic mean daily return / annualised volatility
- Risk-free rate = 0
- Maximum drawdown = largest realised peak-to-trough decline

Do not treat CAGR and the arithmetic mean used in Sharpe as the same quantity.

A positive Sharpe can coexist with a negative CAGR in a highly volatile realised series.

---

## Sentiment Methodology

The baseline model is Standard VADER.

The main innovation is **Finance-Adjusted VADER** using the locked 32-term directional finance lexicon.

Rules:
- Do not overwrite Standard VADER terms
- Preserve raw headline text because VADER uses punctuation, casing, negation and intensifiers
- No-news ticker-days are assigned sentiment = 0
- Interpret 0 on a no-news day as “no new measured news signal”, not objective neutrality
- Sector indices equal-weight the five tickers in each sector
- Trading uses one-trading-day-lagged sector sentiment
- Day t must never use day t sentiment for the trading decision
- Finance-term coverage is limited, so describe the lexicon as a targeted correction rather than a universal sentiment improvement

The fixed sentiment tilt strength is:

`tilt_strength = 1.0`

It was fixed before reviewing the final OOS fusion results and must not be tuned afterward merely to improve performance.

---

## Fusion Baseline

Use **Equity Equal Weight** as the sentiment-fusion baseline.

This isolates the sentiment effect from optimiser noise.

Compare:
- Baseline Equal Weight
- Standard VADER tilt
- Finance-Adjusted VADER tilt

Report negative or null results honestly.

Do not claim Finance-Adjusted VADER beats Equal Weight when it does not.

---

## Coding Rules

When modifying code:

1. Inspect the current file first.
2. Prefer small, auditable changes.
3. Preserve working methodology unless fixing a genuine error.
4. Compile changed Python files before running them.
5. Run the relevant script after compilation.
6. Read the printed output and confirm expected artifacts were produced.
7. Do not silently suppress warnings or failures that affect correctness.
8. Keep reusable logic in `src/` and orchestration/build work in `scripts/`.
9. Use deterministic and reproducible outputs where possible.
10. Never commit raw course data.

Raw data must load through the provided data-access helper.

The deployed Streamlit app should read precomputed artifacts and must not rerun heavy backtests or VADER scoring on load.

---

## Validation Rules

Do not trust plausible output without checking it.

For important results:
- reconcile summary metrics with the underlying return series,
- inspect concentration and portfolio weights,
- verify dates and sample periods,
- confirm annualisation conventions,
- verify sentiment lagging,
- confirm generated figures and tables match the current data,
- check that report claims match generated outputs.

Known development lesson:
an earlier implementation accidentally reapplied monthly target weights every day. The output looked plausible, but auditing exposed the issue. The corrected engine now uses natural weight drift between monthly rebalances.

This is a reminder to audit whether the code matches the intended economic process, not just whether it runs.

---

## Innovation Rules

The primary innovation story is Finance-Adjusted VADER.

Treat it as one deep extension:

Part A vocabulary gap  
→ conservative 32-term lexicon  
→ verified headline-level rescues  
→ term-frequency evidence  
→ sector-index changes  
→ lagged portfolio fusion  
→ honest economic evaluation.

Do not overclaim breadth: only a minority of headlines contain custom finance terms.

Do not tune parameters after seeing OOS results simply to manufacture a stronger result.

Original app features such as the allocation builder and concentration warning can support the project but should not replace the main Finance-Adjusted VADER innovation narrative.

---

## Report and Interpretation Rules

For report work:
- verify every number against project outputs,
- distinguish percentage changes from percentage-point changes,
- reference and interpret every exhibit,
- avoid unsupported causal claims,
- distinguish evidence from inference,
- state limitations openly,
- do not claim predictive power from event validation,
- do not claim sentiment accuracy without a labelled ground-truth benchmark,
- use Australian English,
- avoid marketing hype.

AI may assist with coding, checking and limited drafting/editing support. Final submitted interpretation, reflection, recommendations and assessment decisions must be reviewed by me against the project evidence.

---

## AI Workflow / Transparency

Keep a candid record in `ai/` of important AI-assisted episodes.

For each important episode record:
- task or prompt,
- AI suggestion or output,
- what I checked,
- what was wrong or incomplete,
- what I changed,
- why I changed it,
- final outcome.

Important episodes include:
- identifying and fixing accidental daily-rebalancing behaviour,
- validating the Crypto Minimum Variance metric convention,
- refining the finance lexicon,
- choosing not to tune sentiment strength after observing OOS results,
- verifying real rescued-headline examples,
- building and testing the Streamlit investor journey.

Do not conceal AI use and do not present unverified AI output as fact.
