"""Station 3 - VADER sentiment scoring and sector sentiment indexes.

This module provides:
1. Standard VADER as the baseline sentiment model.
2. Finance-Adjusted VADER as the project innovation.
3. Equal-weight sector sentiment indexes.
4. One-trading-day lagged signals to prevent look-ahead bias.
"""

import re

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Finance-specific directional terms.
#
# These terms are deliberately restricted to words with a reasonably
# clear directional interpretation in financial headlines.
#
# Ambiguous words such as "buy", "sell", "high", "higher",
# "dividend", and "guidance" are not assigned custom scores because
# their interpretation depends heavily on context.
FINANCE_LEXICON = {
    # Positive finance language
    "beat": 2.0,
    "beats": 2.0,
    "outperform": 2.2,
    "outperforms": 2.2,
    "bullish": 2.2,
    "rally": 2.0,
    "rallies": 2.0,
    "rebound": 1.8,
    "rebounds": 1.8,
    "surge": 2.0,
    "surges": 2.0,
    "soar": 2.1,
    "soars": 2.1,
    "upgrade": 2.2,
    "upgraded": 2.2,
    "buyback": 1.5,

    # Negative finance language
    "downgrade": -2.2,
    "downgraded": -2.2,
    "underperform": -2.2,
    "underperforms": -2.2,
    "underweight": -1.8,
    "bearish": -2.2,
    "plunge": -2.5,
    "plunges": -2.5,
    "tumble": -2.0,
    "tumbles": -2.0,
    "slump": -2.0,
    "selloff": -2.3,
    "sell-off": -2.3,
    "downturn": -2.0,
    "default": -2.5,
    "bankruptcy": -3.0,
}


def _finance_term_count(text: str) -> int:
    """Count occurrences of custom finance terms in a headline."""

    tokens = re.findall(
        r"[A-Za-z]+(?:-[A-Za-z]+)?",
        str(text).lower(),
    )

    return sum(
        token in FINANCE_LEXICON
        for token in tokens
    )


def score_headlines(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Score headlines using baseline and finance-adjusted VADER.

    Raw headline text is preserved because VADER uses punctuation,
    capitalisation, intensifiers, and negation.

    Returns both the standard VADER compound score and the custom
    finance-adjusted compound score so the models can be compared
    directly.
    """

    required_columns = {
        "trading_day",
        "ticker",
        "sector",
        "title",
    }

    missing_columns = (
        required_columns
        - set(panel.columns)
    )

    if missing_columns:
        raise ValueError(
            "Headline panel is missing columns: "
            f"{sorted(missing_columns)}"
        )

    df = panel.copy()

    baseline_vader = (
        SentimentIntensityAnalyzer()
    )

    finance_vader = (
        SentimentIntensityAnalyzer()
    )

    finance_vader.lexicon.update(
        FINANCE_LEXICON
    )

    titles = (
        df["title"]
        .fillna("")
        .astype(str)
    )

    df["vader_compound"] = (
        titles.map(
            lambda text:
            baseline_vader
            .polarity_scores(text)[
                "compound"
            ]
        )
    )

    df["finance_vader_compound"] = (
        titles.map(
            lambda text:
            finance_vader
            .polarity_scores(text)[
                "compound"
            ]
        )
    )

    df["finance_term_count"] = (
        titles.map(
            _finance_term_count
        )
    )

    df["contains_finance_term"] = (
        df["finance_term_count"]
        > 0
    )

    df["baseline_neutral"] = (
        df["vader_compound"]
        == 0
    )

    df["finance_neutral"] = (
        df["finance_vader_compound"]
        == 0
    )

    df["neutral_rescued"] = (
        df["baseline_neutral"]
        & ~df["finance_neutral"]
    )

    df["sentiment_change"] = (
        df["finance_vader_compound"]
        - df["vader_compound"]
    )

    return df


def sector_sentiment_index(
    scores: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
    score_column: str = "vader_compound",
) -> pd.DataFrame:
    """Build an equal-weight daily sentiment index by equity sector.

    Method:
    1. Average multiple headlines for each ticker-day.
    2. Treat ticker-days with no headlines as neutral sentiment (0).
    3. Equal-weight the five tickers within each sector.
    4. Lag the usable investment signal by one trading day.

    Parameters
    ----------
    scores:
        Headline-level sentiment output from score_headlines().

    trading_calendar:
        Equity trading calendar.

    score_column:
        Sentiment score to aggregate. Use "vader_compound" for the
        baseline index or "finance_vader_compound" for the innovation.
    """

    required_columns = {
        "trading_day",
        "ticker",
        "sector",
        score_column,
    }

    missing_columns = (
        required_columns
        - set(scores.columns)
    )

    if missing_columns:
        raise ValueError(
            "Sentiment scores are missing columns: "
            f"{sorted(missing_columns)}"
        )

    df = scores.copy()

    df["trading_day"] = pd.to_datetime(
        df["trading_day"]
    )

    calendar = pd.DataFrame(
        {
            "trading_day":
                pd.DatetimeIndex(
                    trading_calendar
                )
        }
    )

    calendar["trading_day"] = (
        pd.to_datetime(
            calendar["trading_day"]
        )
    )

    calendar = (
        calendar
        .drop_duplicates()
        .sort_values(
            "trading_day"
        )
        .reset_index(drop=True)
    )

    ticker_sector = (
        df[
            [
                "ticker",
                "sector",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "ticker"
        )
        .reset_index(drop=True)
    )

    sector_counts = (
        ticker_sector
        .groupby(
            "ticker"
        )["sector"]
        .nunique()
    )

    if (
        sector_counts
        != 1
    ).any():
        raise ValueError(
            "At least one ticker maps "
            "to multiple sectors."
        )

    # Multiple headlines for the same ticker
    # on the same trading day are averaged first.
    ticker_day = (
        df
        .groupby(
            [
                "trading_day",
                "ticker",
                "sector",
            ],
            as_index=False,
        )
        .agg(
            ticker_sentiment=(
                score_column,
                "mean",
            ),
            headline_count=(
                score_column,
                "size",
            ),
        )
    )

    # Create every possible trading-day x ticker
    # combination so no-news observations are explicit.
    calendar["_key"] = 1
    ticker_sector["_key"] = 1

    full_panel = (
        calendar
        .merge(
            ticker_sector,
            on="_key",
        )
        .drop(
            columns="_key"
        )
    )

    ticker_day = (
        full_panel
        .merge(
            ticker_day,
            on=[
                "trading_day",
                "ticker",
                "sector",
            ],
            how="left",
        )
    )

    ticker_day["had_news"] = (
        ticker_day[
            "headline_count"
        ]
        .notna()
    )

    # No new headline means no new measured
    # news signal, so sentiment is neutral.
    ticker_day[
        "ticker_sentiment"
    ] = (
        ticker_day[
            "ticker_sentiment"
        ]
        .fillna(0.0)
    )

    ticker_day[
        "headline_count"
    ] = (
        ticker_day[
            "headline_count"
        ]
        .fillna(0)
        .astype(int)
    )

    # Equal-weight the tickers within each sector.
    sector_daily = (
        ticker_day
        .groupby(
            [
                "trading_day",
                "sector",
            ],
            as_index=False,
        )
        .agg(
            sentiment=(
                "ticker_sentiment",
                "mean",
            ),
            headline_count=(
                "headline_count",
                "sum",
            ),
            ticker_news_count=(
                "had_news",
                "sum",
            ),
            tickers_in_sector=(
                "ticker",
                "nunique",
            ),
        )
    )

    sector_daily[
        "coverage_share"
    ] = (
        sector_daily[
            "ticker_news_count"
        ]
        / sector_daily[
            "tickers_in_sector"
        ]
    )

    sector_daily = (
        sector_daily
        .sort_values(
            [
                "sector",
                "trading_day",
            ]
        )
        .reset_index(drop=True)
    )

    # Critical look-ahead protection.
    # Day t can use only sentiment from
    # trading day t-1 or earlier.
    sector_daily[
        "sentiment_lag1"
    ] = (
        sector_daily
        .groupby(
            "sector"
        )["sentiment"]
        .shift(1)
    )

    sector_daily = (
        sector_daily
        .rename(
            columns={
                "trading_day":
                    "date",
            }
        )
    )

    return sector_daily[
        [
            "date",
            "sector",
            "sentiment",
            "sentiment_lag1",
            "headline_count",
            "ticker_news_count",
            "tickers_in_sector",
            "coverage_share",
        ]
    ]


def sentiment_model_comparison(
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise baseline versus finance-adjusted VADER by sector."""

    required_columns = {
        "sector",
        "vader_compound",
        "finance_vader_compound",
        "baseline_neutral",
        "finance_neutral",
        "neutral_rescued",
        "contains_finance_term",
    }

    missing_columns = (
        required_columns
        - set(scores.columns)
    )

    if missing_columns:
        raise ValueError(
            "Comparison data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    comparison = (
        scores
        .groupby(
            "sector",
            as_index=False,
        )
        .agg(
            headlines=(
                "vader_compound",
                "size",
            ),
            baseline_mean=(
                "vader_compound",
                "mean",
            ),
            finance_mean=(
                "finance_vader_compound",
                "mean",
            ),
            baseline_neutral_rate=(
                "baseline_neutral",
                "mean",
            ),
            finance_neutral_rate=(
                "finance_neutral",
                "mean",
            ),
            neutral_rescue_rate=(
                "neutral_rescued",
                "mean",
            ),
            finance_term_coverage=(
                "contains_finance_term",
                "mean",
            ),
        )
    )

    comparison[
        "neutral_rate_reduction"
    ] = (
        comparison[
            "baseline_neutral_rate"
        ]
        - comparison[
            "finance_neutral_rate"
        ]
    )

    return comparison