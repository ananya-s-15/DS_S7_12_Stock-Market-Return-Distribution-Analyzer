
import pandas as pd
import numpy as np
from scipy.stats import norm


def calculate_var_cvar(returns, confidence=0.95):

    alpha = 1 - confidence
    results = []

    for ticker in returns.columns:

        series = returns[ticker].dropna()

        if series.empty:
            continue

        cutoff = series.quantile(alpha)

        historical_var = -cutoff

        tail_returns = series[series <= cutoff]
        historical_cvar = -tail_returns.mean()

        mean = series.mean()
        std = series.std()

        parametric_var = -(
            mean + std * norm.ppf(alpha)
        )

        in_sample_breach = (
            series < cutoff
        ).mean()

        results.append({
            "Ticker": ticker,
            "Historical VaR": historical_var,
            "Historical CVaR": historical_cvar,
            "Parametric VaR": parametric_var,
            "In-sample Breach Rate": in_sample_breach
        })

    return pd.DataFrame(results).set_index("Ticker")


def backtest_historical_var(
    returns,
    confidence=0.95,
    window=252
):
    """
    Estimate VaR using the previous 252 observations
    and test it against the next day's actual return.
    """

    alpha = 1 - confidence
    results = []

    for ticker in returns.columns:

        series = returns[ticker].dropna()

        if len(series) <= window:
            continue

        # Use only previous observations to estimate VaR
        thresholds = (
            series.shift(1)
            .rolling(window=window)
            .quantile(alpha)
        )

        comparison = pd.concat(
            [series.rename("Actual Return"),
             thresholds.rename("VaR Threshold")],
            axis=1
        ).dropna()

        if comparison.empty:
            continue

        breaches = (
            comparison["Actual Return"]
            < comparison["VaR Threshold"]
        )

        observed_rate = breaches.mean()
        expected_rate = alpha

        results.append({
            "Ticker": ticker,
            "Test Observations": len(comparison),
            "Exceptions": int(breaches.sum()),
            "Expected Breach Rate": expected_rate,
            "Observed Breach Rate": observed_rate,
            "Breach Rate Difference": (
                observed_rate - expected_rate
            )
        })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).set_index("Ticker")


def calculate_max_drawdown(returns):

    results = {}

    for ticker in returns.columns:

        series = returns[ticker].dropna()

        if series.empty:
            results[ticker] = np.nan
            continue

        wealth = (1 + series).cumprod()
        peak = wealth.cummax()

        drawdown = wealth / peak - 1

        results[ticker] = drawdown.min()

    return pd.Series(results, name="Max Drawdown")


def holding_period_loss_probability(
    returns,
    horizons=(1, 21, 63, 126, 252, 756)
):
    """
    Calculate historical probability of a negative
    compounded return for each holding period.
    """

    results = {}

    for horizon in horizons:

        probabilities = {}

        for ticker in returns.columns:

            series = returns[ticker].dropna()

            if len(series) < horizon:
                probabilities[ticker] = np.nan
                continue

            compounded_returns = (
                (1 + series)
                .rolling(window=horizon)
                .apply(np.prod, raw=True) - 1
            )

            # Exclude unavailable rolling windows
            valid_returns = compounded_returns.dropna()

            if valid_returns.empty:
                probabilities[ticker] = np.nan
            else:
                probabilities[ticker] = (
                    valid_returns < 0
                ).mean()

        results[f"{horizon} days"] = probabilities

    return pd.DataFrame(results).T

def calculate_portfolio_analysis(returns):
    """
    Analyze an equal-weighted portfolio.

    The portfolio is rebalanced to equal weights each day.
    Dates with missing stock returns are excluded so that
    the same stock universe is used throughout.
    """

    clean_returns = returns.dropna(how="any")

    if clean_returns.empty:
        raise ValueError(
            "No common dates available for portfolio analysis."
        )

    if clean_returns.shape[1] < 2:
        raise ValueError(
            "At least two stocks are required."
        )

    # Equal weight for every stock
    weights = np.repeat(
        1 / clean_returns.shape[1],
        clean_returns.shape[1]
    )

    # Daily rebalanced portfolio returns
    portfolio_returns = clean_returns.dot(weights)

    # Portfolio growth, starting at 1
    portfolio_growth = (
        1 + portfolio_returns
    ).cumprod()

    annualized_return = (
        portfolio_growth.iloc[-1]
        ** (252 / len(portfolio_returns)) - 1
    )

    annualized_volatility = (
        portfolio_returns.std() * np.sqrt(252)
    )

    annualized_mean_return = portfolio_returns.mean() * 252

    sharpe_ratio = (
        annualized_mean_return / annualized_volatility
        if annualized_volatility > 0
        else np.nan
    )

    historical_var = -portfolio_returns.quantile(0.05)

    historical_cvar = -portfolio_returns[
        portfolio_returns <= portfolio_returns.quantile(0.05)
    ].mean()

    wealth_peak = portfolio_growth.cummax()

    max_drawdown = (
        portfolio_growth / wealth_peak - 1
    ).min()

    portfolio_metrics = {
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio (0% Risk-Free)": sharpe_ratio,
        "Historical VaR (95%)": historical_var,
        "Historical CVaR (95%)": historical_cvar,
        "Maximum Drawdown": max_drawdown
    }

    correlation_matrix = clean_returns.corr()

    return (
        portfolio_returns,
        portfolio_growth,
        portfolio_metrics,
        correlation_matrix
    )
