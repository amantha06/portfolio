#Stores each symbol’s OHLCV in data/SYMBOL.csv
#Adjust for splits/dividends automatically

import os
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()  # expects API keys in .env if used

def fetch_ohlcv(symbol: str, period="5y", interval="1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    #symbols = ["SPY","QQQ","IWM"]
    symbols = ["SPY"]
    for sym in symbols:
        df = fetch_ohlcv(sym)
        df.to_csv(f"data/{sym}.csv")
        print(f"Saved data/{sym}.csv")
