"""
Risk metrics: Sharpe ratio, Sortino ratio, Calmar ratio, max drawdown,
Value at Risk, Conditional VaR, and beta.
"""

import numpy as np
import pandas as pd

from config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR
from validation import require_columns


def sharpe_ratio(df: pd.DataFrame, return_col: str = "Daily_Return",
                  risk_free: float = RISK_FREE_RATE) -> float:
    """Annualized Sharpe ratio from daily returns."""
    require_columns(df, [return_col], "sharpe_ratio")
    excess_daily_rf = risk_free / TRADING_DAYS_PER_YEAR
    excess_returns = df[return_col].dropna() - excess_daily_rf
    if excess_returns.std() == 0:
        return np.nan
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sortino_ratio(df: pd.DataFrame, return_col: str = "Daily_Return",
                   risk_free: float = RISK_FREE_RATE) -> float:
    """Annualized Sortino ratio — like Sharpe but only penalizes downside volatility.

    Textbook downside deviation: square the *shortfall below the target* for
    every observation (zero for observations that beat the target), then take
    the mean over ALL observations (not just the negative ones) before
    square-rooting. Averaging over only the negative subset — as the previous
    version did — systematically overstates downside deviation and understates
    the ratio, because it drops all the zero-shortfall days from the average
    instead of counting them as zero.
    """
    require_columns(df, [return_col], "sortino_ratio")
    excess_daily_rf = risk_free / TRADING_DAYS_PER_YEAR
    excess_returns = df[return_col].dropna() - excess_daily_rf

    downside_sq = np.minimum(excess_returns, 0) ** 2
    downside_deviation = np.sqrt(downside_sq.mean())

    if downside_deviation == 0 or np.isnan(downside_deviation):
        return np.nan
    return (excess_returns.mean() / downside_deviation) * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(df: pd.DataFrame, price_col: str = "Adj_Close") -> float:
    """Largest peak-to-trough decline, expressed as a negative fraction (e.g. -0.35)."""
    require_columns(df, [price_col], "max_drawdown")
    cumulative_max = df[price_col].cummax()
    drawdown = (df[price_col] - cumulative_max) / cumulative_max
    return drawdown.min()


def drawdown_series(df: pd.DataFrame, price_col: str = "Adj_Close") -> pd.Series:
    """Full drawdown series (for plotting), aligned to df's index."""
    require_columns(df, [price_col], "drawdown_series")
    cumulative_max = df[price_col].cummax()
    return (df[price_col] - cumulative_max) / cumulative_max


def calmar_ratio(df: pd.DataFrame, price_col: str = "Adj_Close",
                  return_col: str = "Daily_Return") -> float:
    """Annualized return divided by the absolute max drawdown.

    Measures return earned per unit of the worst historical loss — a common
    complement to Sharpe/Sortino since it's driven by a single tail event
    rather than the full return distribution.
    """
    from analytics import annualized_return  # local import avoids a circular import at module load

    require_columns(df, [price_col, return_col], "calmar_ratio")
    mdd = max_drawdown(df, price_col)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return annualized_return(df, return_col) / abs(mdd)


def value_at_risk(df: pd.DataFrame, return_col: str = "Daily_Return",
                   confidence: float = 0.95) -> float:
    """Historical (empirical) daily Value at Risk at the given confidence level.
    Returns a negative number representing the daily loss threshold."""
    require_columns(df, [return_col], "value_at_risk")
    return df[return_col].dropna().quantile(1 - confidence)


def conditional_var(df: pd.DataFrame, return_col: str = "Daily_Return",
                     confidence: float = 0.95) -> float:
    """Expected Shortfall / CVaR: average loss in the worst (1 - confidence) tail."""
    require_columns(df, [return_col], "conditional_var")
    var = value_at_risk(df, return_col, confidence)
    tail_losses = df[return_col].dropna()
    tail_losses = tail_losses[tail_losses <= var]
    return tail_losses.mean() if len(tail_losses) > 0 else np.nan


def beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """Beta of a stock relative to a market/benchmark return series (aligned by index)."""
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    if len(aligned) < 2:
        return np.nan
    covariance = aligned.cov().iloc[0, 1]
    market_var = aligned.iloc[:, 1].var()
    return covariance / market_var if market_var != 0 else np.nan


def risk_summary(df: pd.DataFrame, ticker: str) -> dict:
    """One-row dict of all key risk metrics for a ticker."""
    require_columns(df, ["Daily_Return", "Adj_Close"], "risk_summary")
    return {
        "Ticker": ticker,
        "Sharpe_Ratio": sharpe_ratio(df),
        "Sortino_Ratio": sortino_ratio(df),
        "Calmar_Ratio": calmar_ratio(df),
        "Max_Drawdown": max_drawdown(df),
        "VaR_95": value_at_risk(df, confidence=0.95),
        "CVaR_95": conditional_var(df, confidence=0.95),
    }
