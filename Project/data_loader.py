
import yfinance as yf
import pandas as pd
from pathlib import Path


# Stocks selected for MarketPulse analysis
TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "ITC.NS",
    "LT.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "HINDUNILVR.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "BAJFINANCE.NS",
    "ASIANPAINT.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "^NSEI"
]


def download_prices(start="2015-01-01"):

    all_prices = {}

    for ticker in TICKERS:
        print(f"Downloading {ticker}...")

        try:
            data = yf.download(
                ticker,
                start=start,
                auto_adjust=True,
                progress=False
            )

            if data.empty:
                print(f"No data available for {ticker}")
                continue

            close_prices = data["Close"]

            if isinstance(close_prices, pd.DataFrame):
                close_prices = close_prices.iloc[:, 0]

            all_prices[ticker] = close_prices

        except Exception as error:
            print(f"Error downloading {ticker}: {error}")

    if not all_prices:
        raise RuntimeError("No stock data was downloaded.")

    prices = pd.DataFrame(all_prices)
    prices.dropna(how="all", inplace=True)

    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "data"
    data_dir.mkdir(exist_ok=True)

    file_path = data_dir / "stock_prices.csv"
    prices.to_csv(file_path)

    print("\nDownload completed!")
    print(f"Saved file: {file_path}")
    print(f"Dataset shape: {prices.shape}")

    return prices
