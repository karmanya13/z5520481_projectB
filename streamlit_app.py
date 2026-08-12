"""No Bull Investing — FINS3645 FinTech Project Part B.

The deployed app reads precomputed artefacts from results/.
It does not rerun portfolio backtests or sentiment scoring.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.ticker import PercentFormatter


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
RESULTS_DATA = PROJECT_ROOT / "results" / "data"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"


st.set_page_config(
    page_title="No Bull Investing",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_performance_metrics() -> pd.DataFrame:
    """Load the precomputed 12-fund performance table."""

    path = RESULTS_TABLES / "performance_metrics.csv"

    if not path.exists():
        raise FileNotFoundError(
            "Missing results/tables/performance_metrics.csv. "
            "Run scripts/run_part_b.py before launching the app."
        )

    return pd.read_csv(
        path,
        parse_dates=["first_live_date"],
    )


@st.cache_data(show_spinner=False)
def load_fund_returns() -> pd.DataFrame:
    """Load precomputed daily fund returns and growth paths."""

    path = RESULTS_DATA / "fund_returns.csv"

    if not path.exists():
        raise FileNotFoundError(
            "Missing results/data/fund_returns.csv. "
            "Run scripts/run_part_b.py before launching the app."
        )

    return pd.read_csv(
        path,
        parse_dates=["date"],
    )


@st.cache_data(show_spinner=False)
def load_fund_weights() -> pd.DataFrame:
    """Load precomputed monthly target portfolio weights."""

    path = RESULTS_DATA / "fund_weights.csv"

    if not path.exists():
        raise FileNotFoundError(
            "Missing results/data/fund_weights.csv. "
            "Run scripts/run_part_b.py before launching the app."
        )

    return pd.read_csv(
        path,
        parse_dates=["date"],
    )


@st.cache_data(show_spinner=False)
def load_sentiment_comparison() -> pd.DataFrame:
    """Load precomputed Standard vs Finance-Adjusted VADER diagnostics."""

    path = RESULTS_TABLES / "sentiment_model_comparison.csv"

    if not path.exists():
        raise FileNotFoundError(
            "Missing results/tables/sentiment_model_comparison.csv. "
            "Run scripts/run_part_b.py before launching the app."
        )

    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_sector_sentiment() -> pd.DataFrame:
    """Load the precomputed sector sentiment index."""

    path = RESULTS_DATA / "sector_sentiment_index.csv"

    if not path.exists():
        raise FileNotFoundError(
            "Missing results/data/sector_sentiment_index.csv. "
            "Run scripts/run_part_b.py before launching the app."
        )

    return pd.read_csv(
        path,
        parse_dates=["date"],
    )


def format_metrics_table(data: pd.DataFrame) -> pd.DataFrame:
    """Create a clean investor-facing metrics table."""

    display = data[
        [
            "fund",
            "annualised_return",
            "annualised_volatility",
            "sharpe",
            "max_drawdown",
            "final_growth_1",
        ]
    ].copy()

    display = display.rename(
        columns={
            "fund": "Fund",
            "annualised_return": "Annualised Return (CAGR)",
            "annualised_volatility": "Annualised Volatility",
            "sharpe": "Sharpe",
            "max_drawdown": "Max Drawdown",
            "final_growth_1": "Final Growth of $1",
        }
    )

    return display


def make_return_risk_chart(data: pd.DataFrame):
    """Create an interactive-page return-risk comparison chart."""

    family_markers = {
        "Equity": "o",
        "Combined": "s",
        "Crypto": "^",
    }

    method_short = {
        "Equal Weight": "EW",
        "Minimum Variance": "MV",
        "Maximum Sharpe": "MS",
        "Risk Parity": "RP",
    }

    family_short = {
        "Equity": "Eq",
        "Combined": "Comb",
        "Crypto": "Cry",
    }

    fig, ax = plt.subplots(
        figsize=(9.5, 5.5)
    )

    for family in [
        "Equity",
        "Combined",
        "Crypto",
    ]:
        family_data = data[
            data["family"] == family
        ]

        if family_data.empty:
            continue

        ax.scatter(
            family_data["annualised_volatility"],
            family_data["annualised_return"],
            s=90,
            marker=family_markers[family],
            label=family,
        )

        for row in family_data.itertuples(
            index=False
        ):
            label = (
                f"{family_short[row.family]} "
                f"{method_short[row.method_label]}"
            )

            label_offsets = {
                "Eq EW": (-34, 8),
                "Eq MV": (-45, -2),
                "Eq MS": (8, -8),
                "Eq RP": (-42, 10),
                "Comb EW": (8, 8),
                "Comb MV": (8, -12),
                "Comb MS": (8, 8),
                "Comb RP": (8, 8),
                "Cry EW": (8, 6),
                "Cry MV": (8, 6),
                "Cry MS": (8, -2),
                "Cry RP": (8, 6),
            }

            ax.annotate(
                label,
                (
                    row.annualised_volatility,
                    row.annualised_return,
                ),
                xytext=label_offsets.get(label, (6, 6)),
                textcoords="offset points",
                fontsize=8,
            )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
        alpha=0.5,
    )

    ax.set_xlabel(
        "Annualised volatility"
    )

    ax.set_ylabel(
        "Annualised return (CAGR)"
    )

    ax.xaxis.set_major_formatter(
        PercentFormatter(
            xmax=1.0,
            decimals=0,
        )
    )

    ax.yaxis.set_major_formatter(
        PercentFormatter(
            xmax=1.0,
            decimals=0,
        )
    )

    ax.legend(
        title="Asset family",
        frameon=False,
    )

    ax.grid(
        axis="both",
        alpha=0.15,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    return fig



def make_fund_growth_chart(data: pd.DataFrame, fund_name: str):
    """Create a growth-of-$1 chart for one selected fund."""

    fund = (
        data[
            data["fund"] == fund_name
        ]
        .copy()
        .sort_values("date")
    )

    if fund.empty:
        raise ValueError(
            f"No fund return observations found for {fund_name}."
        )

    fig, ax = plt.subplots(
        figsize=(9.5, 4.8)
    )

    ax.plot(
        fund["date"],
        fund["growth_1"],
        linewidth=2.2,
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
        alpha=0.5,
    )

    final_row = fund.iloc[-1]

    ax.annotate(
        f"${final_row['growth_1']:.2f}",
        (
            final_row["date"],
            final_row["growth_1"],
        ),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        fontweight="bold",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value ($)")

    ax.grid(
        axis="y",
        alpha=0.15,
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    return fig


def calculate_user_allocation(
    fund_returns: pd.DataFrame,
    allocations: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    """Build a user allocation from precomputed fund return series.

    The chosen starting allocation is invested once at the beginning of the
    common sample and then allowed to drift naturally. No daily fund-level
    rebalancing is imposed.
    """

    positive_allocations = {
        fund: weight
        for fund, weight in allocations.items()
        if weight > 0
    }

    if not positive_allocations:
        raise ValueError(
            "At least one fund must receive a positive allocation."
        )

    weights = pd.Series(
        positive_allocations,
        dtype=float,
    )

    weights = weights / weights.sum()

    selected = fund_returns[
        fund_returns["fund"].isin(
            positive_allocations.keys()
        )
    ][
        [
            "date",
            "fund",
            "return",
        ]
    ].copy()

    pivot = selected.pivot(
        index="date",
        columns="fund",
        values="return",
    )

    pivot = (
        pivot[
            list(positive_allocations.keys())
        ]
        .dropna()
        .sort_index()
    )

    if pivot.empty:
        raise ValueError(
            "The selected funds do not share a common out-of-sample period."
        )

    # Start each fund sleeve with the user's chosen capital allocation.
    sleeve_values = (
        (1.0 + pivot)
        .cumprod()
        .mul(
            weights,
            axis=1,
        )
    )

    portfolio_value = sleeve_values.sum(
        axis=1
    )

    portfolio_return = (
        portfolio_value
        .pct_change()
    )

    # The first observation includes the first realised return from each fund.
    # Use it directly from the weighted fund returns rather than dropping it.
    first_return = (
        pivot.iloc[0]
        * weights
    ).sum()

    portfolio_return.iloc[0] = (
        first_return
    )

    result = pd.DataFrame(
        {
            "date": portfolio_value.index,
            "portfolio_return":
                portfolio_return.values,
            "growth_1":
                portfolio_value.values,
        }
    )

    all_crypto = all(
        fund.startswith("Crypto ")
        for fund in positive_allocations
    )

    periods_per_year = (
        365
        if all_crypto
        else 252
    )

    r = (
        result["portfolio_return"]
        .dropna()
    )

    growth = (
        result["growth_1"]
    )

    annualised_return = (
        growth.iloc[-1]
        ** (
            periods_per_year
            / len(r)
        )
        - 1.0
    )

    annualised_volatility = (
        r.std(ddof=1)
        * periods_per_year ** 0.5
    )

    annualised_mean_return = (
        r.mean()
        * periods_per_year
    )

    sharpe = (
        annualised_mean_return
        / annualised_volatility
        if annualised_volatility > 0
        else float("nan")
    )

    running_peak = growth.cummax()

    drawdown = (
        growth
        / running_peak
        - 1.0
    )

    max_drawdown = float(
        drawdown.min()
    )

    metrics = {
        "annualised_return":
            float(annualised_return),
        "annualised_volatility":
            float(annualised_volatility),
        "sharpe":
            float(sharpe),
        "max_drawdown":
            max_drawdown,
        "final_growth_1":
            float(growth.iloc[-1]),
        "start_date":
            result["date"].min(),
        "end_date":
            result["date"].max(),
        "annualisation":
            periods_per_year,
    }

    return result, metrics


def make_allocation_growth_chart(
    allocation_path: pd.DataFrame,
):
    """Create a growth-of-$1 chart for a user-created fund allocation."""

    fig, ax = plt.subplots(
        figsize=(9.5, 4.8)
    )

    ax.plot(
        allocation_path["date"],
        allocation_path["growth_1"],
        linewidth=2.2,
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
        alpha=0.5,
    )

    final_row = (
        allocation_path.iloc[-1]
    )

    ax.annotate(
        f"${final_row['growth_1']:.2f}",
        (
            final_row["date"],
            final_row["growth_1"],
        ),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        fontweight="bold",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value ($)")

    ax.grid(
        axis="y",
        alpha=0.15,
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    return fig


def make_sector_sentiment_chart(
    data: pd.DataFrame,
    sector: str,
):
    """Compare Standard and Finance-Adjusted sentiment for one sector."""

    sector_data = (
        data[
            data["sector"] == sector
        ]
        .copy()
        .sort_values("date")
    )

    if sector_data.empty:
        raise ValueError(
            f"No sentiment data found for sector {sector}."
        )

    sector_data["vader_rolling_20"] = (
        sector_data["vader_sentiment"]
        .rolling(
            20,
            min_periods=1,
        )
        .mean()
    )

    sector_data["finance_rolling_20"] = (
        sector_data["finance_sentiment"]
        .rolling(
            20,
            min_periods=1,
        )
        .mean()
    )

    fig, ax = plt.subplots(
        figsize=(9.5, 4.8)
    )

    ax.plot(
        sector_data["date"],
        sector_data["vader_rolling_20"],
        linewidth=2.0,
        label="Standard VADER",
    )

    ax.plot(
        sector_data["date"],
        sector_data["finance_rolling_20"],
        linewidth=2.0,
        label="Finance-Adjusted VADER",
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
        alpha=0.5,
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment index value")

    ax.legend(
        frameon=False,
        loc="upper right",
    )

    ax.grid(
        axis="y",
        alpha=0.15,
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    return fig


def make_sector_coverage_chart(
    data: pd.DataFrame,
    sector: str,
):
    """Plot news coverage share for one sector."""

    sector_data = (
        data[
            data["sector"] == sector
        ]
        .copy()
        .sort_values("date")
    )

    if sector_data.empty:
        raise ValueError(
            f"No coverage data found for sector {sector}."
        )

    sector_data["coverage_20"] = (
        sector_data["coverage_share"]
        .rolling(
            20,
            min_periods=1,
        )
        .mean()
    )

    fig, ax = plt.subplots(
        figsize=(9.5, 2.8)
    )

    ax.plot(
        sector_data["date"],
        sector_data["coverage_20"],
        linewidth=1.8,
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Ticker news coverage")

    ax.yaxis.set_major_formatter(
        PercentFormatter(
            xmax=1.0,
            decimals=0,
        )
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.grid(
        axis="y",
        alpha=0.15,
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    return fig

def main() -> None:
    """Render the investor-facing application."""

    st.title("No Bull Investing")

    st.caption(
        "Systematically managed multi-asset funds for investors who want "
        "transparent portfolio choices without building optimisation models themselves."
    )

    try:
        metrics = load_performance_metrics()
        fund_returns = load_fund_returns()
        fund_weights = load_fund_weights()
        sentiment_comparison = load_sentiment_comparison()
        sector_sentiment = load_sector_sentiment()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    tab_compare, tab_fact_sheet, tab_allocate, tab_sentiment = st.tabs(
        [
            "Compare Funds",
            "Fund Fact Sheet",
            "Build an Allocation",
            "Sentiment Analytics",
        ]
    )

    with tab_compare:
        st.subheader("Compare the 12 available funds")

        st.write(
            "Compare Equity, Crypto and Combined strategies across return, "
            "risk, Sharpe ratio and drawdown. All results are walk-forward "
            "out-of-sample backtests."
        )

        family_options = [
            "All",
            "Equity",
            "Combined",
            "Crypto",
        ]

        selected_family = st.segmented_control(
            "Asset family",
            options=family_options,
            default="All",
        )

        if selected_family in {
            "Equity",
            "Combined",
            "Crypto",
        }:
            filtered = metrics[
                metrics["family"]
                == selected_family
            ].copy()
        else:
            filtered = metrics.copy()

        if filtered.empty:
            st.warning(
                "No funds match the selected filter."
            )
            st.stop()

        best_return = filtered.loc[
            filtered[
                "annualised_return"
            ].idxmax()
        ]

        best_sharpe = filtered.loc[
            filtered[
                "sharpe"
            ].idxmax()
        ]

        lowest_vol = filtered.loc[
            filtered[
                "annualised_volatility"
            ].idxmin()
        ]

        card_1, card_2, card_3 = st.columns(3)

        with card_1:
            st.metric(
                "Highest CAGR",
                f"{best_return['annualised_return']:.1%}",
            )
            st.caption(best_return["fund"])

        with card_2:
            st.metric(
                "Highest Sharpe",
                f"{best_sharpe['sharpe']:.2f}",
            )
            st.caption(best_sharpe["fund"])

        with card_3:
            st.metric(
                "Lowest Volatility",
                f"{lowest_vol['annualised_volatility']:.1%}",
            )
            st.caption(lowest_vol["fund"])

        st.markdown("#### Return versus risk")

        chart = make_return_risk_chart(
            filtered
        )

        st.pyplot(
            chart,
            width="stretch",
        )

        plt.close(chart)

        st.markdown("#### Performance metrics")

        table = format_metrics_table(
            filtered
            .sort_values(
                [
                    "family",
                    "method_label",
                ]
            )
        )

        st.dataframe(
            table.style.format(
                {
                    "Annualised Return (CAGR)": "{:.2%}",
                    "Annualised Volatility": "{:.2%}",
                    "Sharpe": "{:.3f}",
                    "Max Drawdown": "{:.2%}",
                    "Final Growth of $1": "${:.3f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        with st.expander(
            "How these metrics are calculated"
        ):
            st.write(
                "Annualised return is reported as CAGR from the realised "
                "out-of-sample growth path. Sharpe uses the annualised "
                "arithmetic mean of daily returns divided by annualised "
                "volatility, with a risk-free rate of zero."
            )

            st.write(
                "Funds rebalance monthly using a 252-observation rolling "
                "estimation window. Portfolio holdings drift naturally between "
                "monthly rebalances. Equity and Combined funds use 252 trading "
                "days for annualisation; Crypto uses 365."
            )

    with tab_fact_sheet:
        st.subheader("Fund Fact Sheet")

        st.write(
            "Choose any fund to inspect its realised out-of-sample growth, "
            "risk profile and latest target holdings."
        )

        fund_options = (
            metrics["fund"]
            .sort_values()
            .tolist()
        )

        selected_fund = st.selectbox(
            "Select a fund",
            options=fund_options,
            index=fund_options.index(
                "Combined Risk Parity"
            )
            if "Combined Risk Parity" in fund_options
            else 0,
        )

        selected_metrics = (
            metrics[
                metrics["fund"] == selected_fund
            ]
            .iloc[0]
        )

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)

        with metric_1:
            st.metric(
                "CAGR",
                f"{selected_metrics['annualised_return']:.2%}",
            )

        with metric_2:
            st.metric(
                "Volatility",
                f"{selected_metrics['annualised_volatility']:.2%}",
            )

        with metric_3:
            st.metric(
                "Sharpe",
                f"{selected_metrics['sharpe']:.3f}",
            )

        with metric_4:
            st.metric(
                "Max Drawdown",
                f"{selected_metrics['max_drawdown']:.2%}",
            )

        st.markdown("#### Growth of $1")

        growth_chart = make_fund_growth_chart(
            fund_returns,
            selected_fund,
        )

        st.pyplot(
            growth_chart,
            width="stretch",
        )

        plt.close(growth_chart)

        selected_weights = (
            fund_weights[
                fund_weights["fund"]
                == selected_fund
            ]
            .copy()
        )

        if selected_weights.empty:
            st.warning(
                "No portfolio weights are available for this fund."
            )
        else:
            latest_date = (
                selected_weights["date"]
                .max()
            )

            latest_weights = (
                selected_weights[
                    selected_weights["date"]
                    == latest_date
                ]
                .copy()
                .sort_values(
                    "weight",
                    ascending=False,
                )
            )

            largest_weight = float(
                latest_weights[
                    "weight"
                ].max()
            )

            st.markdown("#### Latest target holdings")

            st.caption(
                "Monthly rebalance target weights as at "
                f"{latest_date.strftime('%d %b %Y')}."
            )

            if largest_weight >= 0.40:
                st.error(
                    "High concentration: the largest current position is "
                    f"{largest_weight:.1%} of the fund."
                )
            elif largest_weight >= 0.20:
                st.warning(
                    "Moderate concentration: the largest current position is "
                    f"{largest_weight:.1%} of the fund."
                )
            else:
                st.success(
                    "Diversified current allocation: the largest position is "
                    f"{largest_weight:.1%} of the fund."
                )

            holdings_display = (
                latest_weights[
                    [
                        "ticker",
                        "weight",
                    ]
                ]
                .rename(
                    columns={
                        "ticker": "Asset",
                        "weight": "Target Weight",
                    }
                )
                .head(12)
            )

            st.dataframe(
                holdings_display.style.format(
                    {
                        "Target Weight": "{:.2%}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

            if len(latest_weights) > 12:
                st.caption(
                    f"Showing the 12 largest positions out of "
                    f"{len(latest_weights)} assets."
                )

        with st.expander(
            "How to interpret this fact sheet"
        ):
            st.write(
                "CAGR measures the compound annual growth rate of the realised "
                "out-of-sample portfolio path. Volatility measures annualised "
                "return variability, Sharpe compares the annualised arithmetic "
                "mean return with volatility, and maximum drawdown measures the "
                "largest fall from a previous portfolio peak."
            )

            st.write(
                "Latest holdings are target weights set at the most recent "
                "monthly rebalance. Actual holdings drift between rebalances "
                "as asset prices change."
            )

    with tab_allocate:
        st.subheader("Build an Allocation")

        st.write(
            "Choose up to four funds, assign a starting percentage to each, "
            "and see how that allocation would have behaved over their common "
            "out-of-sample period."
        )

        fund_options = (
            metrics["fund"]
            .sort_values()
            .tolist()
        )

        default_funds = [
            fund
            for fund in [
                "Combined Equal Weight",
                "Combined Risk Parity",
            ]
            if fund in fund_options
        ]

        selected_funds = st.multiselect(
            "Choose funds",
            options=fund_options,
            default=default_funds,
            max_selections=4,
        )

        if not selected_funds:
            st.info(
                "Choose at least one fund to build an allocation."
            )
        else:
            st.markdown("#### Starting allocation")

            allocation_columns = st.columns(
                len(selected_funds)
            )

            default_weight = (
                100.0
                / len(selected_funds)
            )

            user_allocations = {}

            for column, fund in zip(
                allocation_columns,
                selected_funds,
            ):
                with column:
                    user_allocations[fund] = st.number_input(
                        fund,
                        min_value=0.0,
                        max_value=100.0,
                        value=float(default_weight),
                        step=5.0,
                        format="%.1f",
                        key=f"allocation_{fund}",
                    )

            total_allocation = sum(
                user_allocations.values()
            )

            st.metric(
                "Total allocation",
                f"{total_allocation:.1f}%",
            )

            if abs(
                total_allocation - 100.0
            ) > 0.01:
                st.warning(
                    "Allocations must sum to exactly 100% before the "
                    "historical portfolio can be calculated."
                )
            else:
                try:
                    allocation_path, allocation_metrics = (
                        calculate_user_allocation(
                            fund_returns,
                            user_allocations,
                        )
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

                    with metric_1:
                        st.metric(
                            "CAGR",
                            f"{allocation_metrics['annualised_return']:.2%}",
                        )

                    with metric_2:
                        st.metric(
                            "Volatility",
                            f"{allocation_metrics['annualised_volatility']:.2%}",
                        )

                    with metric_3:
                        st.metric(
                            "Sharpe",
                            f"{allocation_metrics['sharpe']:.3f}",
                        )

                    with metric_4:
                        st.metric(
                            "Max Drawdown",
                            f"{allocation_metrics['max_drawdown']:.2%}",
                        )

                    st.markdown("#### Historical growth of $1")

                    allocation_chart = make_allocation_growth_chart(
                        allocation_path
                    )

                    st.pyplot(
                        allocation_chart,
                        width="stretch",
                    )

                    plt.close(
                        allocation_chart
                    )

                    st.caption(
                        "Common out-of-sample period: "
                        f"{allocation_metrics['start_date'].strftime('%d %b %Y')} "
                        "to "
                        f"{allocation_metrics['end_date'].strftime('%d %b %Y')}. "
                        "The starting allocation is invested once and fund sleeves "
                        "then drift naturally rather than being reset every day."
                    )

                    with st.expander(
                        "Allocation methodology"
                    ):
                        st.write(
                            "This tool blends the already-computed daily returns "
                            "of the selected funds. It does not rerun the underlying "
                            "asset backtests. The chosen percentages are treated as "
                            "the starting capital split and each fund sleeve is then "
                            "allowed to grow naturally through time."
                        )

                        st.write(
                            "If every selected fund is Crypto, metrics use 365-day "
                            "annualisation. If Equity or Combined funds are included, "
                            "the common trading-day sample uses 252-day annualisation. "
                            "Annualised return is CAGR, while Sharpe uses the "
                            "annualised arithmetic mean of daily portfolio returns."
                        )

    with tab_sentiment:
        st.subheader("Sentiment Analytics")

        st.write(
            "Compare plain VADER with the Finance-Adjusted extension and "
            "explore how the measured news signal changes across sectors."
        )

        total_headlines = int(
            sentiment_comparison[
                "headlines"
            ].sum()
        )

        baseline_neutral_rate = float(
            (
                sentiment_comparison[
                    "baseline_neutral_rate"
                ]
                * sentiment_comparison[
                    "headlines"
                ]
            ).sum()
            / total_headlines
        )

        finance_neutral_rate = float(
            (
                sentiment_comparison[
                    "finance_neutral_rate"
                ]
                * sentiment_comparison[
                    "headlines"
                ]
            ).sum()
            / total_headlines
        )

        finance_term_coverage = float(
            (
                sentiment_comparison[
                    "finance_term_coverage"
                ]
                * sentiment_comparison[
                    "headlines"
                ]
            ).sum()
            / total_headlines
        )

        neutral_rate_reduction = (
            baseline_neutral_rate
            - finance_neutral_rate
        )

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)

        with metric_1:
            st.metric(
                "Mapped headlines",
                f"{total_headlines:,}",
            )

        with metric_2:
            st.metric(
                "Standard neutral rate",
                f"{baseline_neutral_rate:.2%}",
            )

        with metric_3:
            st.metric(
                "Finance-adjusted neutral rate",
                f"{finance_neutral_rate:.2%}",
            )

        with metric_4:
            st.metric(
                "Neutral-rate reduction",
                f"{neutral_rate_reduction * 100:.2f} pp",
            )

        st.caption(
            "The custom finance terms appear in "
            f"{finance_term_coverage:.2%} of mapped headlines. "
            "This is a targeted vocabulary correction rather than a "
            "replacement for the broader VADER model."
        )

        sector_options = (
            sector_sentiment[
                "sector"
            ]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        selected_sector = st.selectbox(
            "Choose a sector",
            options=sector_options,
            index=(
                sector_options.index("Tech")
                if "Tech" in sector_options
                else 0
            ),
            key="sentiment_sector",
        )

        selected_summary = (
            sentiment_comparison[
                sentiment_comparison["sector"]
                == selected_sector
            ]
            .iloc[0]
        )

        sector_1, sector_2, sector_3 = st.columns(3)

        with sector_1:
            st.metric(
                "Sector headlines",
                f"{int(selected_summary['headlines']):,}",
            )

        with sector_2:
            st.metric(
                "Finance-term coverage",
                f"{selected_summary['finance_term_coverage']:.2%}",
            )

        with sector_3:
            st.metric(
                "Neutral-rate reduction",
                f"{selected_summary['neutral_rate_reduction'] * 100:.2f} pp",
            )

        st.markdown(
            f"#### {selected_sector} sentiment over time"
        )

        sentiment_chart = make_sector_sentiment_chart(
            sector_sentiment,
            selected_sector,
        )

        st.pyplot(
            sentiment_chart,
            width="stretch",
        )

        plt.close(
            sentiment_chart
        )

        st.caption(
            "20-trading-day rolling means are shown for readability. "
            "The trading overlay uses the one-trading-day-lagged sector "
            "signal, not the contemporaneous values shown here."
        )

        st.markdown(
            "#### News coverage"
        )

        coverage_chart = make_sector_coverage_chart(
            sector_sentiment,
            selected_sector,
        )

        st.pyplot(
            coverage_chart,
            width="stretch",
        )

        plt.close(
            coverage_chart
        )

        selected_coverage = (
            sector_sentiment[
                sector_sentiment["sector"]
                == selected_sector
            ]["coverage_share"]
            .mean()
        )

        st.caption(
            f"Average ticker-level news coverage for {selected_sector}: "
            f"{selected_coverage:.1%}. "
            "Ticker-days with no headline are assigned neutral sentiment = 0, "
            "so low-coverage days naturally pull the sector index toward zero."
        )

        st.markdown(
            "#### Sector-level model comparison"
        )

        comparison_display = (
            sentiment_comparison[
                [
                    "sector",
                    "headlines",
                    "baseline_neutral_rate",
                    "finance_neutral_rate",
                    "finance_term_coverage",
                    "neutral_rate_reduction",
                ]
            ]
            .rename(
                columns={
                    "sector": "Sector",
                    "headlines": "Headlines",
                    "baseline_neutral_rate": "Standard Neutral",
                    "finance_neutral_rate": "Finance Neutral",
                    "finance_term_coverage": "Finance-Term Coverage",
                    "neutral_rate_reduction": "Neutral Reduction (pp)",
                }
            )
        )

        st.dataframe(
            comparison_display.style.format(
                {
                    "Headlines": "{:,.0f}",
                    "Standard Neutral": "{:.2%}",
                    "Finance Neutral": "{:.2%}",
                    "Finance-Term Coverage": "{:.2%}",
                    "Neutral Reduction (pp)": lambda x: f"{x * 100:.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        with st.expander(
            "How the sentiment index is built"
        ):
            st.write(
                "Each headline is scored with Standard VADER and again with "
                "Finance-Adjusted VADER. The finance model extends the original "
                "lexicon with 32 directional finance terms that were absent "
                "from Standard VADER."
            )

            st.write(
                "Multiple headlines for a ticker on the same trading day are "
                "averaged first. A complete trading-day by ticker panel is then "
                "constructed, and ticker-days with no new headline receive a "
                "neutral score of zero. The five tickers in each sector are "
                "equal-weighted to form the sector index."
            )

            st.write(
                "Trading decisions use the previous trading day's sector "
                "sentiment, which prevents same-day news from leaking into the "
                "portfolio decision."
            )

    st.divider()

    st.caption(
        "Historical backtests are for analysis only and do not guarantee future performance."
    )


if __name__ == "__main__":
    main()
