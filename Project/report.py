
def generate_risk_report(
    metrics,
    risk_metrics,
    regimes,
    clusters=None,
    var_backtest=None
):

    report = []

    report.append("MARKETPULSE - RISK ANALYSIS REPORT")
    report.append("=" * 50)

    # Volatility analysis
    highest_risk = metrics[
        "Annualized Volatility"
    ].idxmax()

    highest_volatility = metrics.loc[
        highest_risk,
        "Annualized Volatility"
    ]

    report.append("\n1. VOLATILITY ANALYSIS")

    report.append(
        f"Highest volatility stock: {highest_risk}"
    )

    report.append(
        f"Annualized volatility: {highest_volatility:.2%}"
    )

    # VaR analysis
    highest_var = risk_metrics[
        "Historical VaR"
    ].idxmax()

    report.append("\n2. VALUE AT RISK")

    report.append(
        f"Highest historical VaR stock: {highest_var}"
    )

    report.append(
        f"Historical VaR: "
        f"{risk_metrics.loc[highest_var, 'Historical VaR']:.2%}"
    )

    report.append(
        f"Historical CVaR: "
        f"{risk_metrics.loc[highest_var, 'Historical CVaR']:.2%}"
    )

    # Drawdown analysis
    worst_drawdown_stock = risk_metrics[
        "Max Drawdown"
    ].idxmin()

    worst_drawdown = risk_metrics.loc[
        worst_drawdown_stock,
        "Max Drawdown"
    ]

    report.append("\n3. DRAWDOWN ANALYSIS")

    report.append(
        f"Worst historical drawdown: {worst_drawdown_stock}"
    )

    report.append(
        f"Maximum drawdown: {worst_drawdown:.2%}"
    )

    # Market regimes
    valid_regimes = regimes[
        regimes["Regime"] != "Insufficient data"
    ]["Regime"]

    regime_counts = valid_regimes.value_counts(
        normalize=True
    )

    report.append("\n4. MARKET REGIME ANALYSIS")

    if not regime_counts.empty:

        report.append(
            f"Most frequent regime: {regime_counts.idxmax()}"
        )

        for name, percentage in regime_counts.items():

            report.append(
                f"{name}: {percentage:.2%}"
            )

    # Risk DNA
    report.append("\n5. RISK DNA ANALYSIS")

    if clusters is not None:

        report.append(
            f"Number of clusters: "
            f"{clusters['Cluster'].nunique()}"
        )

        for name, count in (
            clusters["Risk Profile"].value_counts().items()
        ):

            report.append(
                f"{name}: {count} stocks"
            )

    # Out-of-sample VaR backtesting
    report.append("\n6. OUT-OF-SAMPLE VaR BACKTESTING")

    if (
        var_backtest is not None
        and not var_backtest.empty
    ):

        average_breach = (
            var_backtest["Observed Breach Rate"].mean()
        )

        expected_breach = (
            var_backtest["Expected Breach Rate"].mean()
        )

        report.append(
            f"Average expected breach rate: {expected_breach:.2%}"
        )

        report.append(
            f"Average observed breach rate: {average_breach:.2%}"
        )

        if average_breach > expected_breach:

            report.append(
                "Observed breaches exceeded the expected rate."
            )

        else:

            report.append(
                "Observed breaches did not exceed the expected rate."
            )

    else:

        report.append(
            "Insufficient data for VaR backtesting."
        )

    report.append("\nIMPORTANT NOTES")

    report.append(
        "All measures describe historical market behaviour."
    )

    report.append(
        "VaR backtesting uses a rolling historical estimation window."
    )

    report.append(
        "Historical analysis does not guarantee future performance."
    )

    report.append(
        "This report is for educational purposes only, "
        "not investment advice."
    )

    return "\n".join(report)
6