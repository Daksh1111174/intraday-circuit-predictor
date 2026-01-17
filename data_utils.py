import yfinance as yf
import pandas as pd
import numpy as np

def get_intraday_features(symbol):
    df = yf.download(
        symbol,
        interval="5m",
        period="5d",
        auto_adjust=False,
        progress=False,
        group_by="column"
    )

    if df.empty:
        return df

    # 🔒 Force single-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.dropna(inplace=True)

    df.index = pd.to_datetime(df.index)

    # ---------------- BASIC FEATURES ----------------
    df['pct_change'] = df['Close'].pct_change() * 100
    df['volume_spike'] = df['Volume'] / df['Volume'].rolling(10).mean()

    # ---------------- DAY-WISE VWAP (SAFE) ----------------
    dates = df.index.date

    cum_vol_price = (
        df['Close'].values * df['Volume'].values
    )
    df['cum_vol_price'] = pd.Series(cum_vol_price).groupby(dates).cumsum().values

    df['cum_volume'] = (
        pd.Series(df['Volume'].values).groupby(dates).cumsum().values
    )

    df['vwap'] = df['cum_vol_price'].values / df['cum_volume'].values

    # 🔒 FORCE SERIES ASSIGNMENT (KEY FIX)
    vwap_dist = (df['Close'].values - df['vwap'].values) / df['vwap'].values
    df['vwap_dist'] = vwap_dist

    # ---------------- VOLATILITY ----------------
    df['volatility'] = (
        pd.Series(df['pct_change'].values)
        .rolling(6)
        .std()
        .values
    )

    # ---------------- CLEANUP ----------------
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df


def create_intraday_target(df, circuit_limit=10):
    if df.empty:
        return df

    dates = df.index.date

    day_open = pd.Series(df['Close'].values).groupby(dates).transform('first').values
    day_close = pd.Series(df['Close'].values).groupby(dates).transform('last').values

    df['day_return'] = (day_close - day_open) / day_open * 100

    df['circuit_target'] = np.where(
        df['day_return'] >= circuit_limit, 1,
        np.where(df['day_return'] <= -circuit_limit, -1, 0)
    )

    return df
