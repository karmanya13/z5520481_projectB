"""Station 3 - look-ahead-safe sentiment fusion for equity funds."""

import numpy as np
import pandas as pd

from src import portfolios


def apply_sentiment(
    base_weights: pd.Series,
    sector_signal: pd.Series,
    sector_map: pd.Series,
    tilt_strength: float = 1.0,
) -> pd.Series:
    """Tilt stock weights using the sentiment of each stock's sector.

    The rule is:

        adjusted weight
        = base weight * (1 + tilt_strength * sector sentiment)

    The adjusted weights are then normalised to sum to one.

    A fixed tilt_strength is used rather than choosing the value that
    performs best out of sample, which avoids tuning the strategy to
    future performance.
    """

    if tilt_strength < 0:
        raise ValueError(
            "tilt_strength must be non-negative"
        )

    if not base_weights.index.equals(
        sector_map.index
    ):
        sector_map = sector_map.reindex(
            base_weights.index
        )

    if sector_map.isna().any():
        missing = (
            sector_map[
                sector_map.isna()
            ]
            .index
            .tolist()
        )

        raise ValueError(
            "Missing sector mapping for: "
            f"{missing}"
        )

    stock_signal = (
        sector_map.map(
            sector_signal
        )
    )

    if stock_signal.isna().any():
        missing_sectors = (
            sector_map[
                stock_signal.isna()
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Missing sentiment for sectors: "
            f"{missing_sectors}"
        )

    multipliers = (
        1.0
        + tilt_strength
        * stock_signal
    )

    # VADER compound scores are bounded between -1 and +1.
    # Clipping protects against a zero or negative multiplier.
    multipliers = multipliers.clip(
        lower=0.0
    )

    adjusted_weights = (
        base_weights
        * multipliers
    )

    if adjusted_weights.sum() <= 0:
        raise RuntimeError(
            "Sentiment tilt produced zero total portfolio weight"
        )

    adjusted_weights = (
        adjusted_weights
        / adjusted_weights.sum()
    )

    return adjusted_weights.astype(
        float
    )


def sentiment_tilt_backtest(
    returns: pd.DataFrame,
    sector_map: pd.Series,
    sector_index: pd.DataFrame,
    signal_column: str,
    tilt_strength: float = 1.0,
    estimation_window: int = 252,
    periods_per_year: int = 252,
) -> dict:
    """Run a monthly equal-weight fund with a lagged sector sentiment tilt.

    The strategy:
    1. Wait for the initial 252-observation window.
    2. Rebalance on the first trading day of each eligible month.
    3. Start from equal stock weights.
    4. Tilt using only the supplied lagged sector sentiment signal.
    5. Allow portfolio weights to drift naturally between rebalances.
    """

    returns = portfolios._validate_returns(
        returns
    )

    required_sentiment_columns = {
        "date",
        "sector",
        signal_column,
    }

    missing_columns = (
        required_sentiment_columns
        - set(sector_index.columns)
    )

    if missing_columns:
        raise ValueError(
            "Sector sentiment index is missing columns: "
            f"{sorted(missing_columns)}"
        )

    sector_map = (
        pd.Series(
            sector_map
        )
        .reindex(
            returns.columns
        )
    )

    if sector_map.isna().any():
        raise ValueError(
            "Every equity ticker must have a sector mapping"
        )

    sentiment_data = (
        sector_index[
            [
                "date",
                "sector",
                signal_column,
            ]
        ]
        .copy()
    )

    sentiment_data["date"] = (
        pd.to_datetime(
            sentiment_data["date"]
        )
    )

    if sentiment_data.duplicated(
        [
            "date",
            "sector",
        ]
    ).any():
        raise ValueError(
            "Duplicate date-sector observations in sentiment index"
        )

    sentiment_wide = (
        sentiment_data
        .pivot(
            index="date",
            columns="sector",
            values=signal_column,
        )
        .sort_index()
    )

    rebalance_dates = (
        portfolios._monthly_rebalance_dates(
            returns.index,
            estimation_window,
        )
    )

    if not rebalance_dates:
        raise ValueError(
            "No eligible rebalance dates were found"
        )

    number_of_assets = (
        returns.shape[1]
    )

    base_weights = pd.Series(
        1.0 / number_of_assets,
        index=returns.columns,
        dtype=float,
    )

    portfolio_return_blocks = []
    weight_records = []

    for i, rebalance_date in enumerate(
        rebalance_dates
    ):

        position = returns.index.get_loc(
            rebalance_date
        )

        estimation_data = returns.iloc[
            position - estimation_window:
            position
        ]

        # Defensive check to preserve the same
        # no-look-ahead structure as the baseline funds.
        if (
            estimation_data.index.max()
            >= rebalance_date
        ):
            raise RuntimeError(
                "Look-ahead detected in estimation window"
            )

        if (
            len(estimation_data)
            != estimation_window
        ):
            raise RuntimeError(
                "Incorrect estimation-window length"
            )

        if rebalance_date not in sentiment_wide.index:
            raise ValueError(
                "No sentiment observation for rebalance date "
                f"{rebalance_date}"
            )

        sector_signal = (
            sentiment_wide
            .loc[
                rebalance_date
            ]
        )

        if sector_signal.isna().any():
            missing_sectors = (
                sector_signal[
                    sector_signal.isna()
                ]
                .index
                .tolist()
            )

            raise ValueError(
                "Missing lagged sentiment on "
                f"{rebalance_date}: "
                f"{missing_sectors}"
            )

        target_weights = apply_sentiment(
            base_weights=base_weights,
            sector_signal=sector_signal,
            sector_map=sector_map,
            tilt_strength=tilt_strength,
        )

        if not np.isclose(
            target_weights.sum(),
            1.0,
            atol=1e-8,
        ):
            raise RuntimeError(
                "Target weights do not sum to one"
            )

        if (
            target_weights
            < -1e-10
        ).any():
            raise RuntimeError(
                "Negative portfolio weights detected"
            )

        if i < len(
            rebalance_dates
        ) - 1:

            next_rebalance = (
                rebalance_dates[
                    i + 1
                ]
            )

            live_returns = (
                returns.loc[
                    (
                        returns.index
                        >= rebalance_date
                    )
                    & (
                        returns.index
                        < next_rebalance
                    )
                ]
            )

        else:

            live_returns = (
                returns.loc[
                    returns.index
                    >= rebalance_date
                ]
            )

        # Begin the month at the target weights.
        # Then allow holdings to drift naturally as assets move.
        current_weights = (
            target_weights.copy()
        )

        block_returns = []

        for date, asset_returns in (
            live_returns.iterrows()
        ):

            portfolio_return = float(
                current_weights
                @ asset_returns
            )

            block_returns.append(
                (
                    date,
                    portfolio_return,
                )
            )

            gross_asset_returns = (
                1.0
                + asset_returns
            )

            current_weights = (
                current_weights
                * gross_asset_returns
            )

            portfolio_gross_return = (
                1.0
                + portfolio_return
            )

            if (
                portfolio_gross_return
                <= 0
            ):
                raise RuntimeError(
                    "Portfolio value became non-positive"
                )

            current_weights = (
                current_weights
                / portfolio_gross_return
            )

        block_series = pd.Series(
            {
                date: value
                for date, value
                in block_returns
            },
            dtype=float,
        )

        portfolio_return_blocks.append(
            block_series
        )

        for ticker, weight in (
            target_weights.items()
        ):

            weight_records.append(
                {
                    "date":
                        rebalance_date,
                    "ticker":
                        ticker,
                    "sector":
                        sector_map.loc[
                            ticker
                        ],
                    "weight":
                        weight,
                    "signal":
                        signal_column,
                    "tilt_strength":
                        tilt_strength,
                }
            )

    daily_returns = (
        pd.concat(
            portfolio_return_blocks
        )
        .sort_index()
    )

    if daily_returns.index.has_duplicates:
        raise RuntimeError(
            "Duplicate portfolio return dates detected"
        )

    growth = (
        1.0
        + daily_returns
    ).cumprod()

    weights = pd.DataFrame(
        weight_records
    )

    metrics = (
        portfolios.performance_metrics(
            daily_returns,
            periods_per_year=periods_per_year,
        )
    )

    return {
        "daily_returns":
            daily_returns,
        "growth":
            growth,
        "weights":
            weights,
        "metrics":
            metrics,
        "rebalance_dates":
            rebalance_dates,
        "first_live_date":
            rebalance_dates[0],
        "estimation_window":
            estimation_window,
        "signal_column":
            signal_column,
        "tilt_strength":
            tilt_strength,
    }