import yfinance as yf
import pandas as pd
import numpy as np

def get_intraday_features(symbol):
    df = yf.download(
        symbol,
        interval="5m",
        period="5d",
        auto_adjust=False,
        progress=False
    )

    df.dropna(inplace=True)

    # Ensure datetime index
    df.index = pd.to_datetime(df.index)

    # --- BASIC FEATURES ---
    df['pct_change'] = df['Close'].pct_change() * 100
    df['volume_spike'] = df['Volume'] / df['Volume'].rolling(10).mean()

    # --- VWAP (DAY-WISE CORRECT METHOD) ---
    df['date'] = df.index.date
    df['cum_vol_price'] = (df['Close'] * df['Volume']).groupby(df['date']).cumsum()
    df['cum_volume'] = df['Volume'].groupby(df['date']).cumsum()

    df['vwap'] = df['cum_vol_price'] / df['cum_volume']
    df['vwap_dist'] = (df['Close'] - df['vwap']) / df['vwap']

    # --- VOLATILITY ---
    df['volatility'] = df['pct_change'].rolling(6).std()

    # --- CLEANUP ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df


def create_intraday_target(df, circuit_limit=10):
    df['date'] = df.index.date

    day_open = df.groupby('date')['Close'].transform('first')
    day_close = df.groupby('date')['Close'].transform('last')

    df['day_return'] = (day_close - day_open) / day_open * 100

    df['circuit_target'] = np.where(
        df['day_return'] >= circuit_limit, 1,
        np.where(df['day_return'] <= -circuit_limit, -1, 0)
    )

    return df
