import yfinance as yf
import pandas as pd
import numpy as np

def get_intraday_features(symbol):
    df = yf.download(
        symbol,
        interval="5m",
        period="5d",
        auto_adjust=False
    )

    df.dropna(inplace=True)

    df['pct_change'] = df['Close'].pct_change() * 100
    df['volume_spike'] = df['Volume'] / df['Volume'].rolling(10).mean()

    df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['vwap_dist'] = (df['Close'] - df['vwap']) / df['vwap']

    df['volatility'] = df['pct_change'].rolling(6).std()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df


def create_intraday_target(df, circuit_limit=10):
    daily_open = df.groupby(df.index.date)['Close'].transform('first')
    daily_close = df.groupby(df.index.date)['Close'].transform('last')

    df['day_return'] = (daily_close - daily_open) / daily_open * 100

    df['circuit_target'] = np.where(
        df['day_return'] >= circuit_limit, 1,
        np.where(df['day_return'] <= -circuit_limit, -1, 0)
    )

    return df
