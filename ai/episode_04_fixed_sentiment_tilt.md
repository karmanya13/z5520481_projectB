# Episode 4 — Keeping the Sentiment Tilt Fixed

## Task

The project combines lagged sector sentiment with the Equity Equal Weight portfolio.

The portfolio weight adjustment uses a fixed sentiment tilt.

## Method

The chosen value was:

`tilt_strength = 1.0`

This was fixed before reviewing the final out-of-sample fusion results.

## Result

The three portfolio results were approximately:

Baseline Equal Weight:
- CAGR 12.64%
- Sharpe 0.819

Standard VADER Tilt:
- CAGR 12.02%
- Sharpe 0.786

Finance-Adjusted VADER Tilt:
- CAGR 12.17%
- Sharpe 0.795

Finance-Adjusted VADER improved on Standard VADER but did not beat the Equal Weight baseline.

## AI Discussion

AI identified that one possible extension would be to search over different tilt strengths to find a better-performing portfolio.

## My Decision

I chose not to optimise the tilt strength after seeing the out-of-sample results.

## Reason

Selecting the parameter that performed best after observing the evaluation period would effectively fit the model to the same period being used to judge performance.

This would make any apparent improvement less credible.

## Final Outcome

The fixed value of 1.0 was retained and the negative portfolio result was reported honestly.

## Reflection

This was an example where not following a potentially performance-enhancing AI suggestion produced a more defensible methodology.

The project was evaluated based on whether the sentiment extension added useful information, not whether a parameter search could manufacture the strongest historical result.
