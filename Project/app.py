
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from pathlib import Path
from scipy import stats
from scipy.stats import norm

from src.data_loader import download_prices
from src.metrics import calculate_returns, calculate_basic_metrics

from src.risk import (
    calculate_var_cvar,
    backtest_historical_var,
    calculate_max_drawdown,
    holding_period_loss_probability,
    calculate_portfolio_analysis
)
from src.regimes import detect_market_regimes
from src.clustering import cluster_risk_profiles
from src.report import generate_risk_report


# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="MarketPulse",
    page_icon="📈",
    layout="wide"
)

st.title("📈 MarketPulse")
st.subheader("Stock Return Distribution & Risk Analytics System")

st.markdown(
    "Analyze historical stock returns, volatility, "
    "market regimes and investment risk."
)


# -----------------------------------
# LOAD DATA
# -----------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "stock_prices.csv"


@st.cache_data
def load_data():

    if DATA_FILE.exists():

        prices = pd.read_csv(
            DATA_FILE,
            index_col=0,
            parse_dates=True
        )

    else:

        prices = download_prices()

    prices = prices.apply(
        pd.to_numeric,
        errors="coerce"
    )

    prices = prices.dropna(how="all")

    return prices


try:

    prices = load_data()

except Exception as error:

    st.error(f"Could not load stock data: {error}")
    st.stop()


# -----------------------------------
# CALCULATE RETURNS AND METRICS
# -----------------------------------

returns = calculate_returns(prices)

market_returns = (
    returns["^NSEI"]
    if "^NSEI" in returns.columns
    else returns.mean(axis=1)
)

stock_returns = returns.drop(
    columns=["^NSEI"],
    errors="ignore"
)

stock_returns = stock_returns.dropna(
    axis=1,
    how="all"
)

basic_metrics = calculate_basic_metrics(
    stock_returns
)


risk_metrics = calculate_var_cvar(stock_returns)

risk_metrics["Max Drawdown"] = calculate_max_drawdown(
    stock_returns
)

var_backtest = backtest_historical_var(
    stock_returns,
    confidence=0.95,
    window=252
)


regimes = detect_market_regimes(
    market_returns
)

loss_probability = holding_period_loss_probability(
    stock_returns
)

# PORTFOLIO ANALYSIS
(
    portfolio_returns,
    portfolio_growth,
    portfolio_metrics,
    correlation_matrix
) = calculate_portfolio_analysis(stock_returns)

try:

    clusters = cluster_risk_profiles(
        stock_returns
    )

except Exception as error:

    clusters = None
    cluster_error = str(error)


# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("MarketPulse Controls")

selected_stock = st.sidebar.selectbox(
    "Select a stock",
    stock_returns.columns.tolist()
)

st.sidebar.caption(
    "Data source: Yahoo Finance"
)

st.sidebar.caption(
    "Analysis uses historical adjusted prices."
)


# -----------------------------------
# DASHBOARD TABS
# -----------------------------------

tabs = st.tabs([
    "Overview",
    "Distribution Analysis",
    "Risk Metrics",
    "Market Regimes",
    "Risk DNA",
    "Loss Probability",
    "Portfolio Analysis",
    "Risk Report"
])


# -----------------------------------
# 1. OVERVIEW
# -----------------------------------

with tabs[0]:

    st.header("Market Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Stocks Analyzed",
        len(stock_returns.columns)
    )

    col2.metric(
        "Trading Days",
        len(stock_returns)
    )

    col3.metric(
        "Selected Stock",
        selected_stock
    )

    st.divider()

    st.subheader("Annualized Volatility Comparison")

    volatility_data = (
        basic_metrics["Annualized Volatility"]
        .sort_values(ascending=False)
        .reset_index()
    )

    volatility_data.columns = [
        "Ticker",
        "Annualized Volatility"
    ]

    fig = px.bar(
        volatility_data,
        x="Ticker",
        y="Annualized Volatility",
        title="Annualized Volatility of Stocks"
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Statistical Summary")

    st.dataframe(
        basic_metrics.style.format("{:.4f}"),
        width="stretch"
    )


# -----------------------------------
# 2. DISTRIBUTION ANALYSIS
# -----------------------------------

with tabs[1]:

    st.header("Stock Return Distribution")

    series = stock_returns[selected_stock].dropna()

    mean = series.mean()
    std = series.std()

    col1, col2, col3 = st.columns(3)

    col1.metric("Mean Daily Return", f"{mean:.4%}")
    col2.metric("Daily Volatility", f"{std:.4%}")
    col3.metric("Observations", len(series))

    st.subheader("Return Histogram with Normal Curve")

    fig = px.histogram(
        x=series,
        nbins=50,
        histnorm="probability density",
        labels={
            "x": "Daily Return",
            "y": "Density"
        }
    )

    x_values = np.linspace(
        series.min(),
        series.max(),
        300
    )

    if std > 0:

        fig.add_scatter(
            x=x_values,
            y=norm.pdf(x_values, mean, std),
            mode="lines",
            name="Normal Distribution"
        )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Box Plot")

    box_fig = px.box(
        y=series,
        labels={"y": "Daily Return"},
        title=f"Return Spread: {selected_stock}"
    )

    st.plotly_chart(
        box_fig,
        width="stretch"
    )

    st.subheader("Q-Q Plot")

    theoretical, ordered = stats.probplot(
        series,
        dist="norm",
        fit=False
    )

    slope, intercept, correlation = stats.probplot(
        series,
        dist="norm"
    )[1]

    qq_fig = px.scatter(
        x=theoretical,
        y=ordered,
        labels={
            "x": "Theoretical Quantiles",
            "y": "Observed Quantiles"
        },
        title="Normal Q-Q Plot"
    )

    qq_fig.add_scatter(
        x=theoretical,
        y=slope * np.array(theoretical) + intercept,
        mode="lines",
        name="Normal Reference"
    )

    st.plotly_chart(
        qq_fig,
        width="stretch"
    )

    st.subheader("Jarque-Bera Normality Test")

    jb = stats.jarque_bera(series)

    st.write(f"Test Statistic: {jb.statistic:.4f}")
    st.write(f"P-value: {jb.pvalue:.3e}")

    if jb.pvalue < 0.05:

        st.warning(
            "The returns show evidence of deviation from normality."
        )

    else:

        st.success(
            "The test does not provide sufficient evidence "
            "against normality."
        )


# -----------------------------------
# 3. RISK METRICS
# -----------------------------------


with tabs[2]:

    st.header("Risk Measurement")

    st.markdown(
        "VaR estimates a loss threshold. CVaR estimates "
        "the average loss in the worst tail of returns."
    )

    st.subheader("Value at Risk and Conditional VaR")

    st.dataframe(
        risk_metrics.style.format("{:.2%}"),
        width="stretch"
    )

    st.subheader("Maximum Drawdown Comparison")

    drawdown_data = (
        risk_metrics["Max Drawdown"]
        .sort_values()
        .reset_index()
    )

    drawdown_data.columns = [
        "Ticker",
        "Maximum Drawdown"
    ]

    fig = px.bar(
        drawdown_data,
        x="Ticker",
        y="Maximum Drawdown",
        title="Historical Maximum Drawdown"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # NEW SECTION STARTS HERE

    st.divider()

    st.subheader("Out-of-Sample VaR Backtesting")

    st.write(
        "VaR is estimated using the previous 252 trading days "
        "and checked against the following day's actual return."
    )

    if not var_backtest.empty:

        st.dataframe(
            var_backtest.style.format({
                "Expected Breach Rate": "{:.2%}",
                "Observed Breach Rate": "{:.2%}",
                "Breach Rate Difference": "{:+.2%}"
            }),
            width="stretch"
        )

        st.caption(
            "At 95% confidence, approximately 5% of observations "
            "are expected to breach the estimated VaR. "
            "Small differences can occur naturally."
        )

    else:

        st.warning(
            "Not enough data for out-of-sample VaR backtesting."
        )

    # NEW SECTION ENDS HERE


# -----------------------------------
# 4. MARKET REGIME DETECTION
# -----------------------------------

with tabs[3]:

    st.header("Market Regime Detection")

    st.write(
        "Market conditions are classified using "
        "21-trading-day rolling annualized volatility."
    )

    regime_fig = px.scatter(
        regimes.reset_index(),
        x="Date",
        y="Rolling Volatility",
        color="Regime",
        title="Market Volatility and Regimes"
    )

    st.plotly_chart(
        regime_fig,
        width="stretch"
    )

    st.subheader("Regime Distribution")

    regime_counts = (
        regimes["Regime"]
        .value_counts()
        .reset_index()
    )

    regime_counts.columns = [
        "Regime",
        "Number of Days"
    ]

    st.dataframe(
        regime_counts,
        width="stretch"
    )


# -----------------------------------
# 5. RISK DNA CLUSTERING
# -----------------------------------

with tabs[4]:

    st.header("Risk DNA - Stock Clustering")

    st.write(
        "Stocks are grouped based on their historical "
        "return and risk characteristics using K-Means."
    )

    if clusters is not None:
        
        fig = px.scatter(
            clusters.reset_index(),
            x="PCA1",
            y="PCA2",
            color="Risk Profile",
            hover_name="Ticker",
            title="Risk DNA Clusters (PCA Visualization)"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        st.subheader("Cluster Characteristics")

        numeric_columns = clusters.select_dtypes(
            include="number"
        ).columns.tolist()

        st.dataframe(
            clusters.style.format(
                "{:.4f}",
                subset=numeric_columns
            ),
            width="stretch"
        )



    else:

        st.warning(
            f"Clustering could not be completed: {cluster_error}"
        )


# -----------------------------------
# 6. HOLDING-PERIOD LOSS PROBABILITY
# -----------------------------------

with tabs[5]:

    st.header("Holding-Period Loss Probability")

    st.write(
        "This heatmap shows how frequently historical "
        "holding-period returns were negative."
    )

    selected_loss = loss_probability[
        [selected_stock]
    ].dropna()

    fig = px.bar(
        x=selected_loss.index,
        y=selected_loss[selected_stock],
        labels={
            "x": "Holding Period",
            "y": "Probability of Loss"
        },
        title=f"Loss Probability: {selected_stock}"
    )

    fig.update_yaxes(tickformat=".0%")

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.subheader("All-Stock Loss Probability Heatmap")

    heatmap_fig = px.imshow(
        loss_probability,
        aspect="auto",
        color_continuous_scale="RdYlGn_r",
        zmin=0,
        zmax=1,
        labels={
            "x": "Stock",
            "y": "Holding Period",
            "color": "Loss Probability"
        }
    )

    heatmap_fig.update_coloraxes(
        colorbar_tickformat=".0%"
    )

    st.plotly_chart(
        heatmap_fig,
        width="stretch"
    )


# -----------------------------------
# 8. AUTOMATED RISK REPORT
# -----------------------------------

with tabs[7]:

    st.header("Automated Risk Report")

    report = generate_risk_report(
    basic_metrics,
    risk_metrics,
    regimes,
    clusters,
    var_backtest
    )

    st.text(report)

    st.download_button(
        label="Download Risk Report",
        data=report,
        file_name="MarketPulse_Risk_Report.txt",
        mime="text/plain"
    )

# -----------------------------------
# 7. PORTFOLIO RISK & DIVERSIFICATION
# -----------------------------------

with tabs[6]:

    st.header("Portfolio Risk & Diversification")

    st.markdown(
        "Analyze how combining multiple stocks affects "
        "historical portfolio performance and risk."
    )

    st.subheader("Equal-Weighted Portfolio")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Stocks in Portfolio",
        stock_returns.shape[1]
    )

    col2.metric(
        "Annualized Return",
        f"{portfolio_metrics['Annualized Return']:.2%}"
    )

    col3.metric(
        "Annualized Volatility",
        f"{portfolio_metrics['Annualized Volatility']:.2%}"
    )

    st.subheader("Portfolio Risk Summary")

    portfolio_table = pd.DataFrame(
        {
            "Metric": list(portfolio_metrics.keys()),
            "Value": list(portfolio_metrics.values())
        }
    )

    st.dataframe(
        portfolio_table.style.format(
            {"Value": "{:.4f}"}
        ),
        width="stretch"
    )

    st.caption(
        "Equal-weighted portfolio, rebalanced daily. "
        "Sharpe ratio assumes a 0% risk-free rate. "
        "Returns exclude transaction costs and taxes."
    )

    st.divider()

    st.subheader("Historical Portfolio Growth")

    growth_fig = px.line(
        x=portfolio_growth.index,
        y=portfolio_growth.values,
        labels={
            "x": "Date",
            "y": "Portfolio Growth (Starting Value = 1)"
        },
        title="Equal-Weighted Portfolio Growth"
    )

    st.plotly_chart(
        growth_fig,
        width="stretch"
    )

    st.divider()

    st.subheader("Stock Correlation Heatmap")

    correlation_fig = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Between Stocks"
    )

    st.plotly_chart(
        correlation_fig,
        width="stretch"
    )

    st.markdown(
        """
        **How to interpret correlation:**

        - Near +1: Stocks tend to move in the same direction.
        - Near 0: Little linear relationship.
        - Near -1: Stocks tend to move in opposite directions.

        Combining less-correlated stocks can help reduce
        portfolio volatility, although diversification
        cannot eliminate market risk.
        """
    )



# -----------------------------------
# FOOTER
# -----------------------------------

st.divider()

st.caption(
    "MarketPulse | Historical Data Analytics "
    "| Educational Use Only - Not Investment Advice"
)
