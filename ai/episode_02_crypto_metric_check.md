# Episode 2 — Crypto Minimum Variance Metric Check

## Task

During review of the twelve-fund performance table, Crypto Minimum Variance appeared to contain a possible inconsistency.

Its reported values were approximately:

- CAGR: 88.48%
- annualised volatility: 66.77%
- Sharpe: 1.283

Multiplying Sharpe by volatility implied an annualised arithmetic mean return of approximately 85.7%, which appeared lower than the reported CAGR.

## Prompt / Question to AI

I asked AI to investigate whether the fund metrics were being calculated from different return series or whether there was a portfolio-engine error.

## AI Suggestion

The initial concern was that the arithmetic mean should not be below the geometric mean for the same return series.

This raised the possibility that the Sharpe calculation or CAGR calculation was inconsistent.

## What I Checked

I checked the underlying Crypto Minimum Variance daily return series directly.

The approximate values were:

- daily arithmetic mean: 0.23473%
- daily geometric mean: 0.17380%

The arithmetic daily mean was correctly above the geometric daily mean.

The difference arose only after annualisation:

- the Sharpe numerator uses the daily arithmetic mean multiplied by 365,
- CAGR compounds the realised portfolio growth through time.

These are different annualisation conventions.

## What I Changed

No portfolio-engine change was required.

Instead, I made the reporting convention explicit:

- annualised return in the performance table means CAGR,
- Sharpe uses the annualised arithmetic daily mean divided by annualised volatility.

## Final Outcome

The Crypto Minimum Variance result was retained.

The report also explains that a positive Sharpe can coexist with poor compounded growth in a highly volatile series, which is particularly relevant to Crypto Maximum Sharpe.

## Reflection

This episode showed the value of checking apparently inconsistent metrics directly against the underlying daily return series before changing working code.
