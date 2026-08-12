# AI Notes

## How I Used AI

I used generative AI throughout Part B mainly to speed up coding, debugging, checking and presentation. The most useful role was having another tool challenge results or suggest places where the implementation could be wrong.

The project was developed iteratively. I usually made one change at a time, ran the relevant Python file locally, inspected the output and then decided whether to keep the change.

This was more useful than asking AI to generate an entire project at once because several outputs initially looked reasonable even when the underlying implementation needed correction.

## What AI Was Good At

AI was particularly useful for:
- identifying possible coding errors,
- helping restructure Python files,
- explaining optimisation and portfolio metrics,
- suggesting validation checks,
- helping improve report-quality figures,
- reviewing whether exhibits met the assignment requirements,
- improving Streamlit layout and investor usability,
- identifying inconsistent terminology or numerical presentation.

For example, AI helped identify that the portfolio implementation was effectively resetting target weights every day rather than allowing holdings to drift between monthly rebalances.

It also helped investigate an apparent inconsistency between Crypto Minimum Variance CAGR and Sharpe, which was resolved by checking the difference between compounded CAGR and the annualised arithmetic mean used in the Sharpe ratio.

## Where AI Was Wrong or Incomplete

AI output was not always reliable.

One important example was the portfolio rebalancing logic. The original implementation produced plausible performance results, but an audit showed that monthly target weights were being reapplied every day. This meant the portfolio was effectively rebalanced daily. I corrected the implementation so weights are set only on monthly rebalance dates and holdings drift naturally between them.

Another example occurred when headline examples were initially suggested for the report. They illustrated the Finance-Adjusted VADER idea but were not verified against the actual dataset. I therefore added `scripts/extract_rescued_examples.py`, reran the real news data through both sentiment models and replaced the illustrative examples with genuine rescued headlines from the project corpus.

AI also initially described the reduction in neutral sentiment from 48.85% to 45.96% as a percentage change. I corrected this to a 2.88 percentage-point reduction.

## Decisions I Made Rather Than Optimising for Results

One important decision was keeping:

`tilt_strength = 1.0`

After seeing that the Finance-Adjusted VADER portfolio did not outperform the Equal Weight baseline, it would have been possible to search for a tilt strength that produced a better historical result.

I chose not to do this because selecting the parameter after observing the out-of-sample performance would reduce the credibility of the test and introduce data-mining.

The negative result was therefore retained in the final analysis.

## How I Checked AI Output

My main checking process was:

1. compile the changed Python file,
2. run the relevant script locally,
3. inspect the printed output,
4. inspect generated CSVs and figures,
5. compare key results against underlying return data where necessary,
6. question results that appeared inconsistent,
7. only then use them in the report or app.

Examples included:
- checking Maximum Sharpe concentration,
- reconciling Crypto Minimum Variance performance,
- checking the one-trading-day sentiment lag,
- verifying the COVID sentiment event,
- validating real rescued headlines,
- testing each Streamlit tab locally.

## Report Use

AI also assisted with drafting, editing and consistency checks during report development. The numerical evidence came from the project outputs rather than from AI-generated estimates. I checked the report against the generated tables and figures and reviewed and revised the final interpretation, limitations and recommendations before submission.

## Overall Reflection

The main lesson from using AI in this project was that plausible output should still be audited.

The most useful AI interactions were not the ones that simply generated code, but the ones that led me to question an assumption, inspect an output or test whether the implementation actually matched the intended methodology.
