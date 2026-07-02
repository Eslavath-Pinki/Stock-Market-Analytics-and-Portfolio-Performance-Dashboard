"""
Core return, volatility, and moving-average calculations.
All functions operate on a single-ticker DataFrame with a 'Date' and 'Adj_Close' column,
or on a wide price DataFrame (columns = tickers) where noted.
"""

import numpy as np
import pandas as pd

from config import TRADING_DAYS_PER_YEAR
from validation import require_columns, require_nonempty


def add_returns(df: pd.DataFrame, price_col: str = "Adj_Close") -> pd.DataFrame:
    """Add daily simple and log returns to a single-ticker price DataFrame."""
    require_nonempty(df, "add_returns")
    require_columns(df, ["Date", price_col], "add_returns")

    df = df.sort_values("Date").copy()
    df["Daily_Return"] = df[price_col].pct_change()
    df["Log_Return"] = np.log(df[price_col] / df[price_col].shift(1))
    return df


def cumulative_return(df: pd.DataFrame, return_col: str = "Daily_Return") -> pd.Series:
    """Growth of $1 invested at the start of the period."""
    require_columns(df, [return_col], "cumulative_return")
    return (1 + df[return_col].fillna(0)).cumprod()


def annualized_return(df: pd.DataFrame, return_col: str = "Daily_Return") -> float:
    """CAGR-style annualized return from daily simple returns."""
    require_columns(df, [return_col], "annualized_return")
    total_growth = (1 + df[return_col].fillna(0)).prod()
    n_days = df[return_col].notna().sum()
    if n_days == 0:
        return np.nan
    years = n_days / TRADING_DAYS_PER_YEAR
    return total_growth ** (1 / years) - 1


def daily_volatility(df: pd.DataFrame, return_col: str = "Daily_Return") -> float:
    """Standard deviation of daily returns."""
    require_columns(df, [return_col], "daily_volatility")
    return df[return_col].std()


def annualized_volatility(df: pd.DataFrame, return_col: str = "Daily_Return") -> float:
    """Daily volatility scaled to an annual figure via sqrt(time)."""
    return daily_volatility(df, return_col) * np.sqrt(TRADING_DAYS_PER_YEAR)


def add_moving_averages(df: pd.DataFrame, price_col: str = "Adj_Close",
                         windows=(20, 50, 200)) -> pd.DataFrame:
    """Add simple moving averages for the given window sizes (in trading days).

    Uses a full window (min_periods == window) so an SMA_200 value only ever
    appears once 200 real observations exist. Earlier rows are NaN, which is
    the correct behavior — a 1-day "200-day average" is not meaningful and
    would silently distort early-history charts/signals otherwise.
    """
    require_nonempty(df, "add_moving_averages")
    require_columns(df, ["Date", price_col], "add_moving_averages")

    df = df.sort_values("Date").copy()
    for w in windows:
        df[f"SMA_{w}"] = df[price_col].rolling(window=w, min_periods=w).mean()
    return df


def add_rolling_volatility(df: pd.DataFrame, return_col: str = "Daily_Return",
                            window: int = 21) -> pd.DataFrame:
    """Add a rolling annualized volatility column (default ~1 trading month)."""
    require_columns(df, ["Date", return_col], "add_rolling_volatility")
    df = df.sort_values("Date").copy()
    df[f"Rolling_Vol_{window}d"] = (
        df[return_col].rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
    return df


def summarize_ticker(df: pd.DataFrame, ticker: str) -> dict:
    """One-row summary dict of key return/volatility stats for a ticker."""
    require_columns(df, ["Daily_Return"], "summarize_ticker")
    return {
        "Ticker": ticker,
        "Annualized_Return": annualized_return(df),
        "Annualized_Volatility": annualized_volatility(df),
        "Total_Return": cumulative_return(df).iloc[-1] - 1,
        "Best_Day": df["Daily_Return"].max(),
        "Worst_Day": df["Daily_Return"].min(),
    }
