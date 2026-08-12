# Episode 1 — Detecting the Rebalancing Bug

## Task

I was reviewing the walk-forward portfolio engine after the initial twelve-fund backtests had been completed.

The intended methodology was monthly rebalancing with natural portfolio-weight drift between rebalance dates.

## Prompt / Question to AI

I asked AI to inspect the portfolio implementation and check whether the monthly rebalance logic was behaving correctly and whether there were any look-ahead or weight-handling problems.

## AI Suggestion / Finding

The review identified that the target monthly weights were effectively being reapplied to daily returns throughout each month.

Although the weights were estimated only once per month, this implementation was equivalent to continuously resetting the portfolio back to the monthly target weights.

The resulting performance curves still looked plausible, which made the problem easy to miss.

## What I Checked

I inspected the portfolio-return logic and compared it with the intended investment process.

The intended process was:

1. calculate new target weights at the monthly rebalance,
2. invest using those weights,
3. allow individual holdings to change in value,
4. allow portfolio weights to drift naturally,
5. reset to new target weights only at the next monthly rebalance.

The existing implementation did not fully reproduce this behaviour.

## What I Changed

I changed the engine so that target weights are applied only at the rebalance date.

Between rebalances, asset holdings are carried forward and updated by each asset's realised return.

Portfolio weights therefore drift naturally until the next monthly rebalance.

## Why I Changed It

Daily resetting would make the realised fund returns inconsistent with the stated monthly-rebalancing methodology.

It could also materially affect portfolio performance, particularly for concentrated portfolios.

## Final Outcome

I reran the full twelve-fund backtest and regenerated the performance metrics and figures using the corrected logic.

All official Part B results use the corrected natural-weight-drift implementation.

## Reflection

This was the most important AI-assisted debugging episode in the project because the incorrect output did not obviously look wrong.

It reinforced the need to audit whether code matches the intended economic process rather than checking only whether the code runs.
