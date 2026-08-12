"""Station 3 - portfolio construction and out-of-sample backtesting."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _validate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Check that the return matrix is ready for portfolio backtesting."""

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if returns.empty:
        raise ValueError("returns is empty")

    out = returns.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out.apply(pd.to_numeric, errors="coerce")

    if out.index.has_duplicates:
        raise ValueError("return matrix contains duplicate dates")
    if out.columns.duplicated().any():
        raise ValueError("return matrix contains duplicate tickers")
    if out.isna().any().any():
        raise ValueError("return matrix contains missing values")
    if (out <= -1.0).any().any():
        raise ValueError("return matrix contains a return <= -100%")

    return out


def _monthly_rebalance_dates(
    index: pd.DatetimeIndex,
    estimation_window: int,
) -> list[pd.Timestamp]:
    """Return the first trading day of each eligible month."""

    dates = pd.Series(index=index, data=index)
    first_days = dates.groupby(index.to_period("M")).first().tolist()

    eligible_dates = []
    for date in first_days:
        date = pd.Timestamp(date)
        position = index.get_loc(date)
        if position >= estimation_window:
            eligible_dates.append(date)

    return eligible_dates


def minimum_variance_weights(estimation_data: pd.DataFrame) -> pd.Series:
    """Calculate long-only minimum-variance portfolio weights."""

    covariance = estimation_data.cov().to_numpy(dtype=float)
    n_assets = estimation_data.shape[1]

    average_variance = float(np.mean(np.diag(covariance)))
    ridge = max(average_variance * 1e-8, 1e-12)
    covariance = covariance + np.eye(n_assets) * ridge

    scale = max(float(np.mean(np.diag(covariance))), 1e-12)
    scaled_covariance = covariance / scale

    initial_weights = np.full(n_assets, 1.0 / n_assets)

    def portfolio_variance(weights):
        return float(weights @ scaled_covariance @ weights)

    constraints = {
        "type": "eq",
        "fun": lambda weights: weights.sum() - 1.0,
    }
    bounds = [(0.0, 1.0) for _ in range(n_assets)]

    result = minimize(
        portfolio_variance,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-10},
    )

    if not result.success:
        raise RuntimeError(
            "Minimum-variance optimisation failed: " f"{result.message}"
        )

    weights = np.clip(result.x, 0.0, None)
    weights = weights / weights.sum()
    return pd.Series(weights, index=estimation_data.columns, dtype=float)


def maximum_sharpe_weights(
    estimation_data: pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.Series:
    """Calculate long-only maximum-Sharpe portfolio weights."""

    mean_returns = estimation_data.mean().to_numpy(dtype=float) * periods_per_year
    covariance = (
        estimation_data.cov().to_numpy(dtype=float) * periods_per_year
    )
    n_assets = estimation_data.shape[1]

    average_variance = float(np.mean(np.diag(covariance)))
    ridge = max(average_variance * 1e-8, 1e-12)
    covariance = covariance + np.eye(n_assets) * ridge

    initial_weights = np.full(n_assets, 1.0 / n_assets)

    def negative_sharpe(weights):
        portfolio_return = float(weights @ mean_returns)
        portfolio_variance = float(weights @ covariance @ weights)
        portfolio_volatility = np.sqrt(max(portfolio_variance, 1e-16))
        return -(portfolio_return / portfolio_volatility)

    constraints = {
        "type": "eq",
        "fun": lambda weights: weights.sum() - 1.0,
    }
    bounds = [(0.0, 1.0) for _ in range(n_assets)]

    result = minimize(
        negative_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-10},
    )

    if not result.success:
        raise RuntimeError(
            "Maximum-Sharpe optimisation failed: " f"{result.message}"
        )

    weights = np.clip(result.x, 0.0, None)
    weights = weights / weights.sum()
    return pd.Series(weights, index=estimation_data.columns, dtype=float)


def risk_parity_weights(estimation_data: pd.DataFrame) -> pd.Series:
    """Calculate long-only risk-parity portfolio weights."""

    covariance = estimation_data.cov().to_numpy(dtype=float)
    n_assets = estimation_data.shape[1]

    average_variance = float(np.mean(np.diag(covariance)))
    ridge = max(average_variance * 1e-8, 1e-12)
    covariance = covariance + np.eye(n_assets) * ridge

    scale = max(float(np.mean(np.diag(covariance))), 1e-12)
    scaled_covariance = covariance / scale

    initial_weights = np.full(n_assets, 1.0 / n_assets)

    def risk_parity_objective(weights):
        portfolio_variance = float(
            weights @ scaled_covariance @ weights
        )
        if portfolio_variance <= 0:
            return 1e10

        marginal_risk = scaled_covariance @ weights
        risk_contributions = weights * marginal_risk
        target_contribution = portfolio_variance / n_assets

        return float(
            np.sum((risk_contributions - target_contribution) ** 2)
        )

    constraints = {
        "type": "eq",
        "fun": lambda weights: weights.sum() - 1.0,
    }
    bounds = [(0.0, 1.0) for _ in range(n_assets)]

    result = minimize(
        risk_parity_objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError(
            "Risk-parity optimisation failed: " f"{result.message}"
        )

    weights = np.clip(result.x, 0.0, None)
    weights = weights / weights.sum()
    return pd.Series(weights, index=estimation_data.columns, dtype=float)


def _buy_and_hold_block_returns(
    live_returns: pd.DataFrame,
    target_weights: pd.Series,
) -> pd.Series:
    """Return fund returns between monthly rebalances with weights allowed to drift.

    The target weights are applied once at the start of the block. Asset holdings then
    move with realised returns until the next rebalance date. This avoids the accidental
    daily rebalancing that would occur if the same target weights were multiplied by
    every day's asset returns.
    """

    holdings = target_weights.reindex(live_returns.columns).astype(float).copy()
    previous_nav = float(holdings.sum())
    portfolio_returns = []

    for date, row in live_returns.iterrows():
        holdings = holdings * (1.0 + row)
        nav = float(holdings.sum())
        portfolio_return = nav / previous_nav - 1.0
        portfolio_returns.append((date, portfolio_return))
        previous_nav = nav

    return pd.Series(
        data=[value for _, value in portfolio_returns],
        index=pd.DatetimeIndex([date for date, _ in portfolio_returns]),
        dtype=float,
        name="return",
    )


def performance_metrics(
    daily_returns: pd.Series,
    periods_per_year: int = 252,
) -> dict:
    """Calculate standard fund performance statistics."""

    r = pd.Series(daily_returns).dropna()
    if r.empty:
        raise ValueError("daily_returns is empty")

    growth = (1.0 + r).cumprod()
    annualised_return = growth.iloc[-1] ** (periods_per_year / len(r)) - 1.0
    annualised_volatility = r.std(ddof=1) * np.sqrt(periods_per_year)
    annualised_mean_return = r.mean() * periods_per_year

    if annualised_volatility > 0:
        sharpe = annualised_mean_return / annualised_volatility
    else:
        sharpe = np.nan

    drawdown = growth / growth.cummax() - 1.0
    max_drawdown = drawdown.min()

    return {
        "annualised_return": annualised_return,
        "annualised_volatility": annualised_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def oos_backtest(
    returns: pd.DataFrame,
    method: str = "equal_weight",
    estimation_window: int = 252,
    periods_per_year: int = 252,
) -> dict:
    """Run a monthly walk-forward out-of-sample backtest.

    Portfolio target weights are formed using information available strictly before
    each rebalance date. The first trading day of each eligible month is used as the
    rebalance date. Target weights are applied on that date and then allowed to drift
    with realised asset returns until the next monthly rebalance.
    """

    returns = _validate_returns(returns)

    allowed_methods = {
        "equal_weight",
        "min_variance",
        "max_sharpe",
        "risk_parity",
    }
    if method not in allowed_methods:
        raise ValueError(f"method must be one of {sorted(allowed_methods)}")
    if estimation_window <= 0:
        raise ValueError("estimation_window must be positive")
    if len(returns) <= estimation_window:
        raise ValueError("Not enough observations for the estimation window")

    rebalance_dates = _monthly_rebalance_dates(
        returns.index,
        estimation_window,
    )
    if not rebalance_dates:
        raise ValueError("No eligible rebalance dates were found")

    portfolio_return_blocks = []
    weight_records = []

    for i, rebalance_date in enumerate(rebalance_dates):
        position = returns.index.get_loc(rebalance_date)
        estimation_data = returns.iloc[
            position - estimation_window : position
        ]

        if estimation_data.index.max() >= rebalance_date:
            raise RuntimeError("Look-ahead detected in estimation window")
        if len(estimation_data) != estimation_window:
            raise RuntimeError("Incorrect estimation-window length")

        if method == "equal_weight":
            portfolio_weights = pd.Series(
                1.0 / returns.shape[1],
                index=returns.columns,
                dtype=float,
            )
        elif method == "min_variance":
            portfolio_weights = minimum_variance_weights(estimation_data)
        elif method == "max_sharpe":
            portfolio_weights = maximum_sharpe_weights(
                estimation_data,
                periods_per_year=periods_per_year,
            )
        else:
            portfolio_weights = risk_parity_weights(estimation_data)

        if portfolio_weights.isna().any():
            raise RuntimeError("Portfolio contains missing weights")
        if (portfolio_weights < -1e-8).any():
            raise RuntimeError("Portfolio contains negative weights")
        if not np.isclose(portfolio_weights.sum(), 1.0, atol=1e-6):
            raise RuntimeError("Portfolio weights do not sum to 1")

        if i < len(rebalance_dates) - 1:
            next_rebalance = rebalance_dates[i + 1]
            live_returns = returns.loc[
                (returns.index >= rebalance_date)
                & (returns.index < next_rebalance)
            ]
        else:
            live_returns = returns.loc[returns.index >= rebalance_date]

        portfolio_returns = _buy_and_hold_block_returns(
            live_returns,
            portfolio_weights,
        )
        portfolio_return_blocks.append(portfolio_returns)

        for ticker, weight in portfolio_weights.items():
            weight_records.append(
                {
                    "date": rebalance_date,
                    "ticker": ticker,
                    "weight": weight,
                    "method": method,
                }
            )

    daily_returns = pd.concat(portfolio_return_blocks).sort_index()
    if daily_returns.index.has_duplicates:
        raise RuntimeError("Backtest produced duplicate portfolio return dates")

    growth = (1.0 + daily_returns).cumprod()
    weights = pd.DataFrame(weight_records)
    metrics = performance_metrics(
        daily_returns,
        periods_per_year=periods_per_year,
    )

    return {
        "daily_returns": daily_returns,
        "growth": growth,
        "weights": weights,
        "metrics": metrics,
        "rebalance_dates": rebalance_dates,
        "first_live_date": rebalance_dates[0],
        "estimation_window": estimation_window,
    }
