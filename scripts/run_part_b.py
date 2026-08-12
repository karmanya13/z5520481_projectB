"""Run the Part B fund backtests and save app-ready outputs.

Run from the project root with:

    python scripts/run_part_b.py
"""

import pathlib
import sys

import pandas as pd

sys.path.insert(
    0,
    str(pathlib.Path(__file__).resolve().parent.parent),
)

from src import etl, features, portfolios, sentiment, fusion  # noqa: E402


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

RESULTS_DATA = PROJECT_ROOT / "results" / "data"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"


ESTIMATION_WINDOW = 252

METHODS = [
    "equal_weight",
    "min_variance",
    "max_sharpe",
    "risk_parity",
]


METHOD_LABELS = {
    "equal_weight": "Equal Weight",
    "min_variance": "Minimum Variance",
    "max_sharpe": "Maximum Sharpe",
    "risk_parity": "Risk Parity",
}


def build_return_matrices():
    """Load clean data and create equity, crypto, and combined return matrices."""

    equities, _ = etl.load_clean_equities()
    crypto, _ = etl.load_clean_crypto()

    equity_returns = features.daily_returns(
        equities,
        asset_class="equity",
    )

    crypto_returns = features.daily_returns(
        crypto,
        asset_class="crypto",
    )

    equity_calendar = etl.equity_trading_calendar(
        equities
    )

    combined_returns = features.combined_returns_panel(
        equity_returns,
        crypto_returns,
        equity_calendar,
    )

    equity_matrix = (
        equity_returns
        .pivot(
            index="date",
            columns="ticker",
            values="ret",
        )
        .sort_index()
        .dropna()
    )

    crypto_matrix = (
        crypto_returns
        .pivot(
            index="date",
            columns="ticker",
            values="ret",
        )
        .sort_index()
        .dropna()
    )

    combined_matrix = (
        combined_returns
        .pivot(
            index="date",
            columns="ticker",
            values="ret",
        )
        .sort_index()
        .dropna()
    )

    return {
        "Equity": {
            "returns": equity_matrix,
            "periods_per_year": 252,
        },
        "Crypto": {
            "returns": crypto_matrix,
            "periods_per_year": 365,
        },
        "Combined": {
            "returns": combined_matrix,
            "periods_per_year": 252,
        },
    }


def run_all_funds(families):
    """Run all asset-family and optimisation-method combinations."""

    return_records = []
    weight_frames = []
    metric_records = []

    total_funds = (
        len(families)
        * len(METHODS)
    )

    counter = 0

    for family, settings in families.items():

        returns = settings["returns"]
        periods_per_year = settings[
            "periods_per_year"
        ]

        for method in METHODS:

            counter += 1

            method_label = METHOD_LABELS[
                method
            ]

            fund_name = (
                f"{family} {method_label}"
            )

            print()
            print(
                f"[{counter}/{total_funds}] "
                f"Running {fund_name}..."
            )

            result = portfolios.oos_backtest(
                returns=returns,
                method=method,
                estimation_window=ESTIMATION_WINDOW,
                periods_per_year=periods_per_year,
            )

            daily_returns = result[
                "daily_returns"
            ]

            growth = result[
                "growth"
            ]

            for date in daily_returns.index:

                return_records.append(
                    {
                        "date": date,
                        "family": family,
                        "method": method,
                        "method_label": method_label,
                        "fund": fund_name,
                        "return": daily_returns.loc[
                            date
                        ],
                        "growth_1": growth.loc[
                            date
                        ],
                    }
                )

            weights = result[
                "weights"
            ].copy()

            weights["family"] = family
            weights["method_label"] = (
                method_label
            )
            weights["fund"] = fund_name

            weights = weights[
                [
                    "date",
                    "family",
                    "method",
                    "method_label",
                    "fund",
                    "ticker",
                    "weight",
                ]
            ]

            weight_frames.append(
                weights
            )

            metrics = result[
                "metrics"
            ]

            metric_records.append(
                {
                    "family": family,
                    "method": method,
                    "method_label": method_label,
                    "fund": fund_name,
                    "first_live_date": result[
                        "first_live_date"
                    ],
                    "estimation_window": result[
                        "estimation_window"
                    ],
                    "periods_per_year": periods_per_year,
                    "annualised_return": metrics[
                        "annualised_return"
                    ],
                    "annualised_volatility": metrics[
                        "annualised_volatility"
                    ],
                    "sharpe": metrics[
                        "sharpe"
                    ],
                    "max_drawdown": metrics[
                        "max_drawdown"
                    ],
                    "final_growth_1": growth.iloc[
                        -1
                    ],
                    "rebalance_count": len(
                        result[
                            "rebalance_dates"
                        ]
                    ),
                }
            )

            print(
                f"    completed: "
                f"return="
                f"{metrics['annualised_return']:.2%}, "
                f"vol="
                f"{metrics['annualised_volatility']:.2%}, "
                f"Sharpe="
                f"{metrics['sharpe']:.2f}"
            )

    fund_returns = pd.DataFrame(
        return_records
    )

    fund_weights = pd.concat(
        weight_frames,
        ignore_index=True,
    )

    performance_metrics = pd.DataFrame(
        metric_records
    )

    return (
        fund_returns,
        fund_weights,
        performance_metrics,
    )


def save_outputs(
    fund_returns,
    fund_weights,
    performance_metrics,
):
    """Save the required Part B fund artifacts."""

    RESULTS_DATA.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_TABLES.mkdir(
        parents=True,
        exist_ok=True,
    )

    fund_returns = (
        fund_returns
        .sort_values(
            [
                "fund",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    fund_weights = (
        fund_weights
        .sort_values(
            [
                "fund",
                "date",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )

    performance_metrics = (
        performance_metrics
        .sort_values(
            [
                "family",
                "method_label",
            ]
        )
        .reset_index(drop=True)
    )

    fund_returns.to_csv(
        RESULTS_DATA
        / "fund_returns.csv",
        index=False,
    )

    fund_weights.to_csv(
        RESULTS_DATA
        / "fund_weights.csv",
        index=False,
    )

    performance_metrics.to_csv(
        RESULTS_TABLES
        / "performance_metrics.csv",
        index=False,
    )

    print()
    print("Saved required fund outputs:")
    print(
        "  results/data/fund_returns.csv"
    )
    print(
        "  results/data/fund_weights.csv"
    )
    print(
        "  results/tables/performance_metrics.csv"
    )


def build_and_save_sentiment():
    """Build and save baseline and finance-adjusted sector sentiment."""

    equities, _ = etl.load_clean_equities()
    news, _ = etl.load_clean_news()

    equity_calendar = etl.equity_trading_calendar(
        equities
    )

    headline_panel = features.assemble_headline_panel(
        news,
        equity_calendar,
    )

    scored_headlines = sentiment.score_headlines(
        headline_panel
    )

    baseline_index = sentiment.sector_sentiment_index(
        scored_headlines,
        equity_calendar,
        score_column="vader_compound",
    )

    finance_index = sentiment.sector_sentiment_index(
        scored_headlines,
        equity_calendar,
        score_column="finance_vader_compound",
    )

    sector_index = baseline_index.rename(
        columns={
            "sentiment": "vader_sentiment",
            "sentiment_lag1": "vader_sentiment_lag1",
        }
    )

    finance_columns = finance_index[
        [
            "date",
            "sector",
            "sentiment",
            "sentiment_lag1",
        ]
    ].rename(
        columns={
            "sentiment": "finance_sentiment",
            "sentiment_lag1": "finance_sentiment_lag1",
        }
    )

    sector_index = sector_index.merge(
        finance_columns,
        on=[
            "date",
            "sector",
        ],
        how="left",
        validate="one_to_one",
    )

    sector_index["sentiment_difference"] = (
        sector_index["finance_sentiment"]
        - sector_index["vader_sentiment"]
    )

    comparison = sentiment.sentiment_model_comparison(
        scored_headlines
    )

    RESULTS_DATA.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_TABLES.mkdir(
        parents=True,
        exist_ok=True,
    )

    sector_index.to_csv(
        RESULTS_DATA
        / "sector_sentiment_index.csv",
        index=False,
    )

    comparison.to_csv(
        RESULTS_TABLES
        / "sentiment_model_comparison.csv",
        index=False,
    )

    print()
    print(
        "Saved sentiment outputs:"
    )
    print(
        "  results/data/sector_sentiment_index.csv"
    )
    print(
        "  results/tables/sentiment_model_comparison.csv"
    )

    print(
        f"  sector-days: {len(sector_index)}"
    )

    print(
        f"  sectors: {sector_index['sector'].nunique()}"
    )

    changed = (
        sector_index["sentiment_difference"]
        .abs()
        > 1e-12
    ).sum()

    print(
        f"  finance-adjusted sector-days changed: "
        f"{changed}"
    )

    return sector_index

def build_and_save_fusion():
    """Run and save the equity sentiment-fusion comparison."""

    equities, _ = etl.load_clean_equities()

    equity_returns = features.daily_returns(
        equities,
        asset_class="equity",
    )

    equity_matrix = (
        equity_returns
        .pivot(
            index="date",
            columns="ticker",
            values="ret",
        )
        .sort_index()
        .dropna()
    )

    sector_map = (
        equities[
            [
                "ticker",
                "sector",
            ]
        ]
        .drop_duplicates()
        .set_index("ticker")["sector"]
    )

    sector_index = pd.read_csv(
        RESULTS_DATA
        / "sector_sentiment_index.csv",
        parse_dates=["date"],
    )

    # Baseline equity equal-weight fund.
    baseline = portfolios.oos_backtest(
        equity_matrix,
        method="equal_weight",
        estimation_window=ESTIMATION_WINDOW,
        periods_per_year=252,
    )

    # Standard VADER tilt.
    vader = fusion.sentiment_tilt_backtest(
        returns=equity_matrix,
        sector_map=sector_map,
        sector_index=sector_index,
        signal_column="vader_sentiment_lag1",
        tilt_strength=1.0,
        estimation_window=ESTIMATION_WINDOW,
        periods_per_year=252,
    )

    # Finance-adjusted VADER tilt.
    finance = fusion.sentiment_tilt_backtest(
        returns=equity_matrix,
        sector_map=sector_map,
        sector_index=sector_index,
        signal_column="finance_sentiment_lag1",
        tilt_strength=1.0,
        estimation_window=ESTIMATION_WINDOW,
        periods_per_year=252,
    )

    strategies = {
        "Baseline Equal Weight": baseline,
        "Standard VADER Tilt": vader,
        "Finance-Adjusted VADER Tilt": finance,
    }

    comparison_records = []
    return_frames = []

    for strategy_name, result in strategies.items():

        metrics = result["metrics"]

        comparison_records.append(
            {
                "strategy": strategy_name,
                "annualised_return": float(
                    metrics["annualised_return"]
                ),
                "annualised_volatility": float(
                    metrics["annualised_volatility"]
                ),
                "sharpe": float(
                    metrics["sharpe"]
                ),
                "max_drawdown": float(
                    metrics["max_drawdown"]
                ),
                "final_growth_1": float(
                    result["growth"].iloc[-1]
                ),
                "first_live_date": result[
                    "first_live_date"
                ],
                "rebalance_count": len(
                    result["rebalance_dates"]
                ),
            }
        )

        strategy_returns = (
            result["daily_returns"]
            .rename("return")
            .reset_index()
        )

        strategy_returns.columns = [
            "date",
            "return",
        ]

        strategy_returns["strategy"] = (
            strategy_name
        )

        strategy_returns["growth_1"] = (
            result["growth"].values
        )

        return_frames.append(
            strategy_returns
        )

    comparison = pd.DataFrame(
        comparison_records
    )

    baseline_row = (
        comparison[
            comparison["strategy"]
            == "Baseline Equal Weight"
        ]
        .iloc[0]
    )

    comparison[
        "return_change_vs_baseline"
    ] = (
        comparison["annualised_return"]
        - baseline_row["annualised_return"]
    )

    comparison[
        "sharpe_change_vs_baseline"
    ] = (
        comparison["sharpe"]
        - baseline_row["sharpe"]
    )

    comparison[
        "drawdown_change_vs_baseline"
    ] = (
        comparison["max_drawdown"]
        - baseline_row["max_drawdown"]
    )

    fusion_returns = pd.concat(
        return_frames,
        ignore_index=True,
    )

    RESULTS_DATA.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_TABLES.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        RESULTS_TABLES
        / "fusion_comparison.csv",
        index=False,
    )

    fusion_returns.to_csv(
        RESULTS_DATA
        / "fusion_returns.csv",
        index=False,
    )

    print()
    print("Saved fusion outputs:")
    print(
        "  results/tables/fusion_comparison.csv"
    )
    print(
        "  results/data/fusion_returns.csv"
    )

    print()
    print(
        comparison[
            [
                "strategy",
                "annualised_return",
                "annualised_volatility",
                "sharpe",
                "max_drawdown",
                "final_growth_1",
            ]
        ].to_string(index=False)
    )

    return comparison
def main():

    print(
        "Building return matrices..."
    )

    families = build_return_matrices()

    for family, settings in families.items():

        matrix = settings[
            "returns"
        ]

        print(
            f"{family}: "
            f"{matrix.shape[0]} dates x "
            f"{matrix.shape[1]} assets"
        )

    print()
    print(
        "Running 12 walk-forward funds..."
    )

    (
        fund_returns,
        fund_weights,
        performance_metrics,
    ) = run_all_funds(
        families
    )

    save_outputs(
        fund_returns,
        fund_weights,
        performance_metrics,
    )

    print()
    print(
        "Performance summary:"
    )

    display_columns = [
        "fund",
        "annualised_return",
        "annualised_volatility",
        "sharpe",
        "max_drawdown",
        "final_growth_1",
    ]

    print(
        performance_metrics[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Building sector sentiment index..."
    )

    build_and_save_sentiment()

    print()
    print(
        "Running sentiment fusion comparison..."
    )

    build_and_save_fusion()

    print()
    print(
        "Part B baseline build complete."
    )


if __name__ == "__main__":
    main()