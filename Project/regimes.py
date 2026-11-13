
import pandas as pd


def detect_market_regimes(market_returns, window=21):

    rolling_volatility = (
        market_returns
        .rolling(window=window)
        .std() * (252 ** 0.5)
    )

    q25 = rolling_volatility.quantile(0.25)
    q75 = rolling_volatility.quantile(0.75)
    q95 = rolling_volatility.quantile(0.95)

    regimes = pd.Series(
        "Insufficient data",
        index=market_returns.index,
        dtype="object"
    )

    valid = rolling_volatility.notna()

    regimes.loc[
        valid & (rolling_volatility <= q25)
    ] = "Calm"

    regimes.loc[
        valid & (rolling_volatility > q25)
        & (rolling_volatility <= q75)
    ] = "Normal"

    regimes.loc[
        valid & (rolling_volatility > q75)
        & (rolling_volatility <= q95)
    ] = "Turbulent"

    regimes.loc[
        valid & (rolling_volatility > q95)
    ] = "Crisis"

    result = pd.DataFrame({
        "Market Return": market_returns,
        "Rolling Volatility": rolling_volatility,
        "Regime": regimes
    })

    return result
