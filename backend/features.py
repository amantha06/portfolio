import os
import pandas as pd
import numpy as np

def load_data(path: str) -> pd.DataFrame:
    """
    Reads the yfinance-dump CSV which has:
      1) header row: Price,Close,High,Low,Open,Volume
      2) ticker row: Ticker,SPY,SPY,...
      3) label row: Date,,,,,
    and then the actual data.

    We skip the first two metadata rows, assign our own column names,
    parse the Date column, coerce to numeric, and drop any NaNs.
    """
    df = pd.read_csv(
        path,
        skiprows=2,             # drop the ticker+label rows
        header=None,
        names=["Date","Close","High","Low","Open","Volume"],
        parse_dates=["Date"],
        infer_datetime_format=True
    )
    df.set_index("Date", inplace=True)

    # Ensure everything but the index is numeric
    for col in ["Close","High","Low","Open","Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a cleaned OHLCV DataFrame, returns a new DataFrame with:
      - r1, r5, r21: lagged returns
      - vol21: 21-day rolling volatility
      - rsi: 14-day RSI
      - macd & macd_signal: 12/26 MACD + 9-day signal
      - bb_upper / bb_lower: 20-day Bollinger Bands (±2σ)
      - atr: 14-day Average True Range
    """
    df = df.copy()

    # 1) Lagged returns & rolling vol
    df["r1"]    = df["Close"].pct_change(1)
    df["r5"]    = df["Close"].pct_change(5)
    df["r21"]   = df["Close"].pct_change(21)
    df["vol21"] = df["r1"].rolling(window=21).std()

    # 2) RSI (14) via Wilder’s smoothing
    delta     = df["Close"].diff()
    gain      = delta.clip(lower=0)
    loss      = -delta.clip(upper=0)
    avg_gain  = gain.rolling(14).mean()
    avg_loss  = loss.rolling(14).mean()
    rs        = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # 3) MACD (12,26) & signal (9)
    ema12             = df["Close"].ewm(span=12, adjust=False).mean()
    ema26             = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # 4) Bollinger Bands (20,2σ)
    m20             = df["Close"].rolling(20).mean()
    sd20            = df["Close"].rolling(20).std()
    df["bb_upper"]  = m20 + 2 * sd20
    df["bb_lower"]  = m20 - 2 * sd20

    # 5) ATR (14)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift(1)).abs()
    tr3 = (df["Low"]  - df["Close"].shift(1)).abs()
    df["tr"]  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(window=14).mean()

    return df.dropna()


if __name__ == "__main__":
    # ensure output folder exists
    os.makedirs("data", exist_ok=True)

    # load & clean raw data
    df_raw = load_data("data/SPY.csv")

    # generate features
    df_feat = add_features(df_raw)

    # save to CSV
    df_feat.to_csv("data/SPY_features_no_talib.csv")
    print(f"Saved {len(df_feat)} rows of features to data/SPY_features_no_talib.csv")
