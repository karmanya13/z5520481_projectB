# Episode 3 — Refining Finance-Adjusted VADER

## Task

Part A showed that Standard VADER frequently failed to recognise finance-specific directional language.

For Part B I wanted to create a more targeted finance-aware sentiment model.

## AI Use

AI was used to help generate candidate finance terms and discuss whether individual words should be assigned positive or negative sentiment scores.

## Initial Approach

An earlier candidate list was broader and contained terms that were either already recognised by VADER or were too context-dependent.

Examples included:

- buy
- sell
- high
- higher
- dividend
- guidance
- miss
- misses

## What I Checked

I compared the proposed terms against Standard VADER's existing vocabulary.

I also considered whether the direction of each word would remain reliable when used in different financial headline contexts.

## What I Changed

I removed ambiguous terms such as:

- buy
- sell
- high
- higher
- dividend
- guidance

I also removed terms already recognised by VADER, including:

- miss
- misses

The final extension contains 32 directional finance terms.

Examples include:

Positive:
- beat
- outperform
- bullish
- rally
- surge
- upgrade
- buyback

Negative:
- downgrade
- underperform
- bearish
- plunge
- sell-off
- downturn
- default
- bankruptcy

## Final Outcome

The final Finance-Adjusted VADER model:

- reduced the exact-neutral headline rate from 48.85% to 45.96%,
- rescued 4,257 previously neutral headlines,
- affected headline scoring selectively rather than across the entire corpus,
- changed approximately 39% of aggregated sector-day index observations.

## Reflection

AI was useful for generating candidate ideas, but the final lexicon required filtering.

A larger vocabulary was not automatically better because ambiguous terms could introduce more classification error than useful financial information.
