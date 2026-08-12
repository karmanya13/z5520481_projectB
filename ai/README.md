# AI Workflow Pack

This folder documents selected examples of how I used generative AI during FINS3645 FinTech Project Part B.

The purpose of this pack is to show where AI assisted, how I checked its suggestions, where outputs were incomplete or wrong, and how I made the final methodological and implementation decisions.

## Overall AI Use

Generative AI was used mainly as a coding, debugging, validation and quality-assurance assistant during Part B. It helped me troubleshoot Python errors, structure code changes, suggest validation checks, review outputs and improve the presentation of figures and Streamlit features.

I retained control over the main methodological decisions, including the twelve-fund structure, monthly rebalancing, the 252-observation rolling estimation window, the Finance-Adjusted VADER extension and the decision to keep the sentiment tilt strength fixed at 1.0 rather than optimising it after observing the results.

AI suggestions were not accepted automatically. I ran the code locally, inspected the generated files and outputs, compared results against the underlying CSVs and corrected issues when something appeared inconsistent.

AI also assisted with drafting and editing during report development. Numerical claims, methodology and exhibits were checked against the actual project outputs, and I reviewed and revised the final interpretation, reflection and recommendations before submission.

Overall, AI accelerated implementation and checking, while the final project decisions, validation and submitted work remained under my direction.

## Curated Workflow Episodes

The following files document representative examples from the project:

1. `episode_01_rebalancing_bug.md`  
   Detecting and fixing accidental daily rebalancing.

2. `episode_02_crypto_metric_check.md`  
   Investigating an apparent inconsistency in Crypto Minimum Variance CAGR and Sharpe.

3. `episode_03_finance_vader.md`  
   Refining the Finance-Adjusted VADER lexicon and removing ambiguous terms.

4. `episode_04_fixed_sentiment_tilt.md`  
   Choosing not to optimise the sentiment tilt after seeing the out-of-sample results.

5. `episode_05_headline_validation.md`  
   Replacing illustrative sentiment examples with verified headlines from the real dataset.

6. `episode_06_streamlit_app.md`  
   Building and testing the investor-facing Streamlit workflow.

Each episode records:
- the task,
- the AI suggestion or output,
- what I checked,
- what was wrong or incomplete,
- what I changed,
- and the final outcome.

## Agent Instructions

The project-level AI instructions are stored at the repository root in:

- `CLAUDE.md`

These instructions contain the methodology, coding rules, validation checks and constraints used while working with AI.

## Principle Used Throughout

AI output was treated as a suggestion to be tested rather than as automatically correct.

Important numerical, methodological and investment conclusions were checked against the code and generated outputs before being accepted.
