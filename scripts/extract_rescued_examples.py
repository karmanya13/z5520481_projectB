"""Extract verified rescued-headline examples and finance-term frequencies.

A rescued headline is one that Standard VADER scores as exactly neutral
(compound == 0) but that becomes directional once the 32-term finance
lexicon is added.

Run from the project root with:

    python scripts/extract_rescued_examples.py

Outputs:
    results/tables/rescued_headline_examples.csv
    results/tables/finance_term_frequency.csv
"""

from __future__ import annotations

import pathlib
import re
import sys

import pandas as pd

sys.path.insert(
    0,
    str(pathlib.Path(__file__).resolve().parent.parent),
)

from src import etl, features, sentiment  # noqa: E402
from src.sentiment import FINANCE_LEXICON  # noqa: E402


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"

N_POSITIVE = 4
N_NEGATIVE = 4


def _matched_terms(title: str) -> list[str]:
    """Return all matched finance terms in headline order."""

    text = str(title).lower()
    found: list[tuple[int, int, str]] = []

    for term in FINANCE_LEXICON:
        pattern = rf"(?<![a-z]){re.escape(term)}(?![a-z])"

        for match in re.finditer(
            pattern,
            text,
        ):
            found.append(
                (
                    match.start(),
                    -len(term),
                    term,
                )
            )

    found.sort()

    ordered_terms: list[str] = []

    for _, _, term in found:
        if term not in ordered_terms:
            ordered_terms.append(term)

    return ordered_terms


def _matched_terms_text(title: str) -> str:
    """Return matched finance terms as a readable comma-separated string."""

    return ", ".join(
        _matched_terms(title)
    )


def _select_varied_examples(
    rescued: pd.DataFrame,
    direction: str,
    n: int,
) -> pd.DataFrame:
    """Select unique positive or negative rescues with sector variety."""

    if direction == "positive":
        candidates = (
            rescued[
                rescued[
                    "finance_vader_compound"
                ] > 0
            ]
            .copy()
            .sort_values(
                "finance_vader_compound",
                ascending=False,
            )
        )
    elif direction == "negative":
        candidates = (
            rescued[
                rescued[
                    "finance_vader_compound"
                ] < 0
            ]
            .copy()
            .sort_values(
                "finance_vader_compound",
                ascending=True,
            )
        )
    else:
        raise ValueError(
            "direction must be 'positive' or 'negative'."
        )

    candidates = (
        candidates
        .drop_duplicates(
            subset="title"
        )
        .copy()
    )

    primary = (
        candidates
        .groupby(
            "sector",
            group_keys=False,
        )
        .head(1)
        .head(n)
    )

    if len(primary) < n:
        extra = (
            candidates[
                ~candidates[
                    "title"
                ].isin(
                    primary["title"]
                )
            ]
            .head(
                n - len(primary)
            )
        )

        primary = pd.concat(
            [
                primary,
                extra,
            ],
            ignore_index=True,
        )

    return primary.head(n)


def main() -> None:
    """Reproduce headline-level rescue evidence from the live corpus."""

    RESULTS_TABLES.mkdir(
        parents=True,
        exist_ok=True,
    )

    equities, _ = etl.load_clean_equities()
    news, _ = etl.load_clean_news()

    calendar = (
        etl.equity_trading_calendar(
            equities
        )
    )

    panel = (
        features.assemble_headline_panel(
            news,
            calendar,
        )
    )

    scored = sentiment.score_headlines(
        panel
    )

    rescued = (
        scored[
            scored[
                "neutral_rescued"
            ]
        ]
        .copy()
    )

    rescued[
        "matched_finance_terms"
    ] = rescued[
        "title"
    ].map(
        _matched_terms_text
    )

    rescued = (
        rescued[
            rescued[
                "matched_finance_terms"
            ] != ""
        ]
        .copy()
    )

    positive = _select_varied_examples(
        rescued,
        "positive",
        N_POSITIVE,
    )

    negative = _select_varied_examples(
        rescued,
        "negative",
        N_NEGATIVE,
    )

    sample = pd.concat(
        [
            positive,
            negative,
        ],
        ignore_index=True,
    )

    examples = sample[
        [
            "title",
            "sector",
            "vader_compound",
            "finance_vader_compound",
            "matched_finance_terms",
        ]
    ].rename(
        columns={
            "title": "headline",
            "vader_compound":
                "standard_compound",
            "finance_vader_compound":
                "finance_compound",
        }
    )

    examples.to_csv(
        RESULTS_TABLES
        / "rescued_headline_examples.csv",
        index=False,
    )

    titles_lower = (
        scored[
            "title"
        ]
        .fillna("")
        .str.lower()
    )

    term_rows = []

    for term in sorted(
        FINANCE_LEXICON
    ):
        pattern = (
            rf"(?<![a-z])"
            rf"{re.escape(term)}"
            rf"(?![a-z])"
        )

        count = int(
            titles_lower
            .str.contains(
                pattern,
                regex=True,
            )
            .sum()
        )

        term_rows.append(
            {
                "term": term,
                "lexicon_score":
                    FINANCE_LEXICON[
                        term
                    ],
                "headline_count":
                    count,
            }
        )

    term_freq = (
        pd.DataFrame(
            term_rows
        )
        .sort_values(
            [
                "headline_count",
                "term",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    term_freq.to_csv(
        RESULTS_TABLES
        / "finance_term_frequency.csv",
        index=False,
    )

    print(
        "Total headlines scored:",
        len(scored),
    )

    print(
        "Rescued "
        "(Standard neutral -> Finance directional):",
        len(rescued),
    )

    print()

    print(
        "Verified examples for report table:"
    )

    print(
        examples.to_string(
            index=False
        )
    )

    print()

    print(
        "Top 10 finance terms by headline count:"
    )

    print(
        term_freq.head(10)
        .to_string(
            index=False
        )
    )

    print()

    print("Saved:")

    print(
        "  results/tables/"
        "rescued_headline_examples.csv"
    )

    print(
        "  results/tables/"
        "finance_term_frequency.csv"
    )


if __name__ == "__main__":
    main()
