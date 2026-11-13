
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def cluster_risk_profiles(returns, n_clusters=4):

    features = []

    for ticker in returns.columns:

        series = returns[ticker].dropna()

        if series.empty:
            continue

        wealth = (1 + series).cumprod()
        drawdown = wealth / wealth.cummax() - 1

        var = -series.quantile(0.05)

        tail = series[
            series <= series.quantile(0.05)
        ]

        cvar = -tail.mean()

        features.append({
            "Ticker": ticker,
            "Annualized Return": (
                series.mean() * 252
            ),
            "Annualized Volatility": (
                series.std() * np.sqrt(252)
            ),
            "Skewness": series.skew(),
            "Kurtosis": series.kurtosis(),
            "VaR": var,
            "CVaR": cvar,
            "Max Drawdown": drawdown.min()
        })

    feature_table = pd.DataFrame(
        features
    ).set_index("Ticker")

    if len(feature_table) < 2:
        raise ValueError(
            "At least two stocks are required."
        )

    feature_table = feature_table.replace(
        [np.inf, -np.inf], np.nan
    )

    feature_table = feature_table.fillna(
        feature_table.median()
    )

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        feature_table
    )

    number_of_clusters = min(
        n_clusters,
        len(feature_table)
    )

    model = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10
    )

    feature_table["Cluster"] = model.fit_predict(
        scaled_features
    )

    # Assign meaningful labels based on cluster risk
    cluster_means = feature_table.groupby(
        "Cluster"
    )[
        ["Annualized Volatility", "Max Drawdown"]
    ].mean()

    risk_score = (
        cluster_means["Annualized Volatility"]
        + cluster_means["Max Drawdown"].abs()
    )

    ordered_clusters = risk_score.sort_values().index

    labels = {
        1: ["Typical Risk"],
        2: ["Lower Risk", "Higher Risk"],
        3: ["Lower Risk", "Moderate Risk", "Higher Risk"],
        4: [
            "Lower Risk",
            "Moderate Risk",
            "Elevated Risk",
            "Higher Risk"
        ]
    }

    names = labels.get(
        number_of_clusters,
        [f"Risk Group {i + 1}"
         for i in range(number_of_clusters)]
    )

    label_mapping = {
        cluster_id: names[position]
        for position, cluster_id in enumerate(
            ordered_clusters
        )
    }

    feature_table["Risk Profile"] = (
        feature_table["Cluster"].map(label_mapping)
    )

    pca = PCA(n_components=2)

    components = pca.fit_transform(
        scaled_features
    )

    feature_table["PCA1"] = components[:, 0]
    feature_table["PCA2"] = components[:, 1]

    return feature_table
