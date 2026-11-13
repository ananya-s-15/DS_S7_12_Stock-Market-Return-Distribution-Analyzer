
import pandas as pd
import numpy as np


def calculate_returns(prices):
    """Calculate daily percentage returns."""
    returns = prices.pct_change(fill_method=None)
    return returns.dropna(how="all")


def calculate_basic_metrics(returns):
    """Calculate descriptive and annualized stock metrics."""

    rows = []

    for ticker in returns.columns:

        series = returns[ticker].dropna()

        if series.empty:
            continue

        daily_mean = series.mean()
        daily_volatility = series.std()

        total_growth = (1 + series).prod()

        cagr = (
            total_growth ** (252 / len(series)) - 1
            if total_growth > 0 else np.nan
        )

        rows.append({
            "Ticker": ticker,
            "Mean Daily Return": daily_mean,
            "Volatility": daily_volatility,
            "Skewness": series.skew(),
            "Kurtosis": series.kurtosis(),
            "Annualized Return": cagr,
            "Annualized Volatility": daily_volatility * np.sqrt(252)
        })

    return pd.DataFrame(rows).set_index("Ticker")
