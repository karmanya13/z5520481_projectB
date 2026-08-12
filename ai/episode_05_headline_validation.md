# Episode 5 — Verifying Rescued Headlines

## Task

I wanted the report to show concrete examples of headlines that Standard VADER classified as neutral but Finance-Adjusted VADER classified directionally.

## Initial AI Output

AI initially suggested example finance headlines that demonstrated how terms such as bullish, beat, downgrade and bankruptcy could change a VADER score.

These examples illustrated the mechanism but had not been verified as actual observations from the project dataset.

## What I Checked

I decided that illustrative examples were not strong enough evidence for the report.

I created and ran:

`scripts/extract_rescued_examples.py`

The script:

1. reloads the real project news data,
2. maps headlines to the equity trading calendar,
3. scores each headline with Standard VADER,
4. scores it again with Finance-Adjusted VADER,
5. selects cases where Standard VADER is exactly neutral and Finance-Adjusted VADER is directional,
6. records the finance terms appearing in each headline,
7. calculates frequency counts for all 32 custom terms.

## Final Verified Results

Total headlines scored: 146,830

Neutral headlines rescued: 4,257

The script produced:

- `results/tables/rescued_headline_examples.csv`
- `results/tables/finance_term_frequency.csv`

The most common finance terms included:

- beat: 2,085 headlines
- rally: 1,065
- beats: 879
- rebound: 680
- surge: 504

## Additional Correction

An early version of the extraction script returned only one triggering term for each headline.

This could be misleading because a headline may contain several finance terms.

I updated the script so it records all matched finance terms.

## Final Outcome

The report now uses only verified headline examples from the actual project corpus.

## Reflection

This episode showed why plausible illustrative evidence should not be treated as empirical evidence until it is reproduced from the underlying dataset.
