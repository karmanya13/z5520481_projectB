"""Create Part B report figures from precomputed results.

Run from the project root with:

    python scripts/make_figures.py
"""

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

RESULTS_DATA = PROJECT_ROOT / "results" / "data"
RESULTS_FIGURES = PROJECT_ROOT / "results" / "figures"

RESULTS_FIGURES.mkdir(
    parents=True,
    exist_ok=True,
)


def make_combined_growth_figure():
    """Plot growth of $1 for the four Combined fund methods."""

    fund_returns = pd.read_csv(
        RESULTS_DATA / "fund_returns.csv",
        parse_dates=["date"],
    )

    combined = (
        fund_returns[
            fund_returns["family"] == "Combined"
        ]
        .copy()
        .sort_values(
            [
                "method_label",
                "date",
            ]
        )
    )

    if combined.empty:
        raise ValueError(
            "No Combined fund observations found."
        )

    methods = [
        "Equal Weight",
        "Minimum Variance",
        "Maximum Sharpe",
        "Risk Parity",
    ]

    found_methods = set(
        combined["method_label"].unique()
    )

    if found_methods != set(methods):
        raise ValueError(
            "Unexpected Combined fund methods. "
            f"Found: {sorted(found_methods)}"
        )

    fig, ax = plt.subplots(
        figsize=(11, 6.5)
    )

    for method in methods:

        method_data = (
            combined[
                combined["method_label"] == method
            ]
            .sort_values("date")
        )

        line = ax.plot(
            method_data["date"],
            method_data["growth_1"],
            linewidth=2.2,
            label=method,
        )[0]

        final_row = method_data.iloc[-1]

        ax.annotate(
            f"${final_row['growth_1']:.2f}",
            xy=(
                final_row["date"],
                final_row["growth_1"],
            ),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=line.get_color(),
        )

    ax.axhline(
        1.0,
        linewidth=1,
        linestyle="--",
        alpha=0.5,
    )

    start_date = (
        combined["date"]
        .min()
        .strftime("%d %b %Y")
    )

    end_date = (
        combined["date"]
        .max()
        .strftime("%d %b %Y")
    )

    ax.set_title(
        "Combined Fund Strategies: Growth of $1",
        fontsize=16,
        fontweight="bold",
        loc="left",
        pad=18,
    )

    ax.text(
        0,
        1.01,
        (
            "Walk-forward out-of-sample performance | "
            f"{start_date} to {end_date}"
        ),
        transform=ax.transAxes,
        fontsize=10,
        alpha=0.75,
    )

    ax.set_xlabel(
        "Date",
        fontsize=11,
    )

    ax.set_ylabel(
        "Portfolio value ($)",
        fontsize=11,
    )

    ax.legend(
        title="Portfolio method",
        frameon=False,
        loc="upper left",
        ncol=2,
    )

    ax.grid(
        axis="y",
        alpha=0.18,
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.margins(
        x=0.05
    )

    fig.text(
        0.11,
        0.02,
        (
            "Note: Monthly rebalancing; 252-observation rolling estimation "
            "window; risk-free rate = 0; transaction costs = 0."
        ),
        fontsize=8.5,
        alpha=0.75,
    )

    fig.tight_layout(
        rect=[
            0,
            0.05,
            1,
            1,
        ]
    )

    output_path = (
        RESULTS_FIGURES
        / "combined_growth_of_1.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved:",
        output_path,
    )


def make_combined_max_sharpe_drawdown():
    """Plot drawdown for the Combined Maximum Sharpe fund."""

    fund_returns = pd.read_csv(
        RESULTS_DATA / "fund_returns.csv",
        parse_dates=["date"],
    )

    fund = (
        fund_returns[
            (
                fund_returns["family"]
                == "Combined"
            )
            & (
                fund_returns["method_label"]
                == "Maximum Sharpe"
            )
        ]
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )

    if fund.empty:
        raise ValueError(
            "Combined Maximum Sharpe observations not found."
        )

    running_peak = (
        fund["growth_1"]
        .cummax()
    )

    fund["drawdown"] = (
        fund["growth_1"]
        / running_peak
        - 1.0
    )

    worst_index = (
        fund["drawdown"]
        .idxmin()
    )

    worst_row = (
        fund.loc[worst_index]
    )

    max_drawdown = (
        worst_row["drawdown"]
    )

    max_drawdown_date = (
        worst_row["date"]
    )

    start_date = (
        fund["date"]
        .min()
        .strftime("%d %b %Y")
    )

    end_date = (
        fund["date"]
        .max()
        .strftime("%d %b %Y")
    )

    fig, ax = plt.subplots(
        figsize=(11, 6.5)
    )

    line = ax.plot(
        fund["date"],
        fund["drawdown"],
        linewidth=2.2,
        label="Drawdown",
    )[0]

    ax.fill_between(
        fund["date"],
        fund["drawdown"],
        0,
        alpha=0.15,
        color=line.get_color(),
    )

    ax.axhline(
        0,
        linewidth=1,
        alpha=0.5,
    )

    ax.scatter(
        max_drawdown_date,
        max_drawdown,
        s=45,
        zorder=5,
        color=line.get_color(),
    )

    ax.annotate(
        (
            f"Worst drawdown: "
            f"{max_drawdown:.1%}"
        ),
        xy=(
            max_drawdown_date,
            max_drawdown,
        ),
        xytext=(
            25,
            40,
        ),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1,
        },
    )

    ax.set_title(
        "Combined Maximum Sharpe: Drawdown",
        fontsize=16,
        fontweight="bold",
        loc="left",
        pad=18,
    )

    ax.text(
        0,
        1.01,
        (
            "Decline from the previous portfolio peak | "
            f"{start_date} to {end_date}"
        ),
        transform=ax.transAxes,
        fontsize=10,
        alpha=0.75,
    )

    ax.set_xlabel(
        "Date",
        fontsize=11,
    )

    ax.set_ylabel(
        "Drawdown",
        fontsize=11,
    )

    ax.yaxis.set_major_formatter(
        PercentFormatter(
            xmax=1.0,
            decimals=0,
        )
    )

    ax.grid(
        axis="y",
        alpha=0.18,
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.text(
        0.11,
        0.02,
        (
            "Note: Drawdown measures the percentage decline from the "
            "fund's previous highest portfolio value."
        ),
        fontsize=8.5,
        alpha=0.75,
    )

    fig.tight_layout(
        rect=[
            0,
            0.05,
            1,
            1,
        ]
    )

    output_path = (
        RESULTS_FIGURES
        / "combined_max_sharpe_drawdown.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved:",
        output_path,
    )

    print(
        "Maximum drawdown:",
        f"{max_drawdown:.2%}",
    )

    print(
        "Maximum drawdown date:",
        max_drawdown_date.date(),
    )


def make_combined_weight_concentration_figure():
    """Plot the largest asset target weight for each Combined method."""

    fund_weights = pd.read_csv(
        RESULTS_DATA / "fund_weights.csv",
        parse_dates=["date"],
    )

    combined = (
        fund_weights[
            fund_weights["family"] == "Combined"
        ]
        .copy()
    )

    if combined.empty:
        raise ValueError(
            "No Combined fund weights found."
        )

    methods = [
        "Equal Weight",
        "Minimum Variance",
        "Maximum Sharpe",
        "Risk Parity",
    ]

    concentration = (
        combined
        .groupby(
            [
                "date",
                "method_label",
            ],
            as_index=False,
        )["weight"]
        .max()
        .rename(
            columns={
                "weight":
                    "largest_asset_weight"
            }
        )
        .sort_values(
            [
                "method_label",
                "date",
            ]
        )
    )

    start_date = (
        concentration["date"]
        .min()
        .strftime("%d %b %Y")
    )

    end_date = (
        concentration["date"]
        .max()
        .strftime("%d %b %Y")
    )

    fig, ax = plt.subplots(
        figsize=(11, 6.5)
    )

    for method in methods:

        method_data = concentration[
            concentration["method_label"]
            == method
        ]

        line = ax.plot(
            method_data["date"],
            method_data[
                "largest_asset_weight"
            ],
            linewidth=2.2,
            marker="o",
            markersize=3,
            label=method,
        )[0]

        final_row = (
            method_data.iloc[-1]
        )

        ax.annotate(
            f"{final_row['largest_asset_weight']:.1%}",
            xy=(
                final_row["date"],
                final_row[
                    "largest_asset_weight"
                ],
            ),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=line.get_color(),
        )

    ax.set_title(
        "Combined Funds: Largest Single-Asset Target Weight",
        fontsize=16,
        fontweight="bold",
        loc="left",
        pad=18,
    )

    ax.text(
        0,
        1.01,
        (
            "Portfolio concentration at each monthly rebalance | "
            f"{start_date} to {end_date}"
        ),
        transform=ax.transAxes,
        fontsize=10,
        alpha=0.75,
    )

    ax.set_xlabel(
        "Rebalance date",
        fontsize=11,
    )

    ax.set_ylabel(
        "Largest asset weight",
        fontsize=11,
    )

    ax.yaxis.set_major_formatter(
        PercentFormatter(
            xmax=1.0,
            decimals=0,
        )
    )

    ax.legend(
        title="Portfolio method",
        frameon=False,
        loc="upper right",
        ncol=2,
    )

    ax.grid(
        axis="y",
        alpha=0.18,
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.margins(
        x=0.05
    )

    fig.text(
        0.11,
        0.02,
        (
            "Note: Values are target weights at monthly rebalances. "
            "A higher value indicates greater concentration in one asset."
        ),
        fontsize=8.5,
        alpha=0.75,
    )

    fig.tight_layout(
        rect=[
            0,
            0.05,
            1,
            1,
        ]
    )

    output_path = (
        RESULTS_FIGURES
        / "combined_weight_concentration.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved:",
        output_path,
    )

    overall_max = (
        concentration
        .loc[
            concentration[
                "largest_asset_weight"
            ].idxmax()
        ]
    )

    print(
        "Highest observed single-asset weight:",
        f"{overall_max['largest_asset_weight']:.2%}",
    )

    print(
        "Method:",
        overall_max["method_label"],
    )

    print(
        "Date:",
        overall_max["date"].date(),
    )


def make_combined_max_sharpe_weights_over_time():
    """Plot the top five Combined Maximum Sharpe target weights over time."""

    fund_weights = pd.read_csv(
        RESULTS_DATA / "fund_weights.csv",
        parse_dates=["date"],
    )

    fund = (
        fund_weights[
            (
                fund_weights["family"]
                == "Combined"
            )
            & (
                fund_weights["method_label"]
                == "Maximum Sharpe"
            )
        ]
        .copy()
        .sort_values(
            [
                "date",
                "ticker",
            ]
        )
    )

    if fund.empty:
        raise ValueError(
            "Combined Maximum Sharpe weights not found."
        )

    weight_sums = (
        fund
        .groupby("date")["weight"]
        .sum()
    )

    if not weight_sums.between(
        0.999999,
        1.000001,
    ).all():
        raise ValueError(
            "Combined Maximum Sharpe weights do not sum to 1."
        )

    average_weights = (
        fund
        .groupby("ticker")["weight"]
        .mean()
        .sort_values(ascending=False)
    )

    top_assets = (
        average_weights
        .head(5)
        .index
        .tolist()
    )

    selected = (
        fund[
            fund["ticker"].isin(top_assets)
        ]
        .copy()
    )

    start_date = (
        fund["date"]
        .min()
        .strftime("%d %b %Y")
    )

    end_date = (
        fund["date"]
        .max()
        .strftime("%d %b %Y")
    )

    fig, ax = plt.subplots(
        figsize=(11, 6.5)
    )

    for ticker in top_assets:

        ticker_data = (
            selected[
                selected["ticker"] == ticker
            ]
            .sort_values("date")
        )

        average_weight = (
            average_weights.loc[ticker]
        )

        ax.plot(
            ticker_data["date"],
            ticker_data["weight"],
            linewidth=2.0,
            marker="o",
            markersize=3,
            label=(
                f"{ticker} "
                f"(avg {average_weight:.1%})"
            ),
        )

    ax.set_title(
        "Combined Maximum Sharpe: Portfolio Weights Over Time",
        fontsize=16,
        fontweight="bold",
        loc="left",
        pad=18,
    )

    ax.text(
        0,
        1.01,
        (
            "Top 5 assets by average target weight | "
            f"{start_date} to {end_date}"
        ),
        transform=ax.transAxes,
        fontsize=10,
        alpha=0.75,
    )

    ax.set_xlabel(
        "Monthly rebalance date",
        fontsize=11,
    )

    ax.set_ylabel(
        "Target portfolio weight",
        fontsize=11,
    )

    ax.yaxis.set_major_formatter(
        PercentFormatter(
            xmax=1.0,
            decimals=0,
        )
    )

    ax.legend(
        title="Asset (average weight)",
        frameon=False,
        loc="upper left",
        ncol=2,
    )

    ax.grid(
        axis="y",
        alpha=0.18,
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.margins(
        x=0.03
    )

    fig.text(
        0.11,
        0.02,
        (
            "Note: Values are target weights set at each monthly rebalance. "
            "The top five assets are selected by average target weight across "
            "the full out-of-sample period; the remaining assets are omitted "
            "for readability."
        ),
        fontsize=8.5,
        alpha=0.75,
    )

    fig.tight_layout(
        rect=[
            0,
            0.06,
            1,
            1,
        ]
    )

    output_path = (
        RESULTS_FIGURES
        / "combined_max_sharpe_weights_over_time.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved:",
        output_path,
    )

    print(
        "Top 5 assets by average target weight:"
    )

    for ticker in top_assets:
        print(
            f"  {ticker}: "
            f"{average_weights.loc[ticker]:.2%}"
        )


def main():

    make_combined_growth_figure()

    make_combined_max_sharpe_drawdown()

    make_combined_weight_concentration_figure()

    make_combined_max_sharpe_weights_over_time()

    print(
        "Figure build complete."
    )


if __name__ == "__main__":
    main()