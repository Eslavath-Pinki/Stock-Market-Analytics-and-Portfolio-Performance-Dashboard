"""
Portfolio construction and multi-stock comparison.
Expects a wide DataFrame of daily returns: index = Date, columns = tickers.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import TRADING_DAYS_PER_YEAR, RISK_FREE_RATE
from validation import require_nonempty


def wide_returns(combined_df: pd.DataFrame, price_col: str = "Adj_Close") -> pd.DataFrame:
    """Pivot a long-format combined price DataFrame into wide daily returns (Date x Ticker)."""
    prices = combined_df.pivot(index="Date", columns="Ticker", values=price_col).sort_index()
    return prices.pct_change().dropna(how="all")


def correlation_matrix(returns_wide: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlation of daily returns across tickers."""
    return returns_wide.corr()


def portfolio_returns(returns_wide: pd.DataFrame, weights: dict) -> pd.Series:
    """Compute the daily return series of a weighted portfolio.
    `weights` maps ticker -> weight (should sum to ~1)."""
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])
    aligned = returns_wide[tickers].dropna()
    return aligned.dot(w)


def portfolio_performance(returns_wide: pd.DataFrame, weights: dict) -> dict:
    """Annualized return, volatility, and Sharpe ratio for a weighted portfolio."""
    port_returns = portfolio_returns(returns_wide, weights)
    ann_return = (1 + port_returns).prod() ** (TRADING_DAYS_PER_YEAR / len(port_returns)) - 1
    ann_vol = port_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    excess = port_returns - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    sharpe = (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS_PER_YEAR) if excess.std() != 0 else np.nan
    return {
        "Annualized_Return": ann_return,
        "Annualized_Volatility": ann_vol,
        "Sharpe_Ratio": sharpe,
    }


def equal_weight_portfolio(tickers: list) -> dict:
    """Convenience helper: equal-weight allocation across the given tickers."""
    w = 1 / len(tickers)
    return {t: w for t in tickers}


def random_portfolios(returns_wide: pd.DataFrame, n_portfolios: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Simulate random portfolios for a basic efficient-frontier scatter plot.
    Returns a DataFrame with columns: Return, Volatility, Sharpe, and one weight column per ticker.

    Vectorized: draws all n_portfolios weight vectors at once and does the
    return/vol/Sharpe math as batched matrix operations instead of a Python
    loop. On a 5-asset, 5000-portfolio simulation this is roughly two orders
    of magnitude faster than looping row by row in pure Python.
    """
    require_nonempty(returns_wide, "random_portfolios")
    rng = np.random.default_rng(seed)
    tickers = returns_wide.columns.tolist()
    n_assets = len(tickers)

    mean_returns = (returns_wide.mean() * TRADING_DAYS_PER_YEAR).values
    cov_matrix = (returns_wide.cov() * TRADING_DAYS_PER_YEAR).values

    weights = rng.random((n_portfolios, n_assets))
    weights /= weights.sum(axis=1, keepdims=True)

    port_returns = weights @ mean_returns
    port_vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov_matrix, weights))
    with np.errstate(invalid="ignore", divide="ignore"):
        sharpes = np.where(port_vols != 0, (port_returns - RISK_FREE_RATE) / port_vols, np.nan)

    df = pd.DataFrame(weights, columns=tickers)
    df.insert(0, "Sharpe", sharpes)
    df.insert(0, "Volatility", port_vols)
    df.insert(0, "Return", port_returns)
    return df


def _portfolio_stats(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray):
    port_return = float(weights @ mean_returns)
    port_vol = float(np.sqrt(weights @ cov_matrix @ weights))
    return port_return, port_vol


def optimize_portfolio(returns_wide: pd.DataFrame, objective: str = "max_sharpe",
                        risk_free: float = RISK_FREE_RATE) -> dict:
    """Find the optimal long-only, fully-invested portfolio via constrained optimization.

    objective:
      - "max_sharpe": maximize the annualized Sharpe ratio
      - "min_vol":    minimize annualized volatility

    Constraints: weights sum to 1 (fully invested, budget constraint) and each
    weight is between 0 and 1 (long-only, no leverage/short-selling) — this is
    the piece the previous version was missing: `scipy` was listed as a
    dependency but never actually used, so there was no real optimizer, only
    a random search that approximates the frontier without ever solving for
    the true optimum.
    """
    require_nonempty(returns_wide, "optimize_portfolio")
    tickers = returns_wide.columns.tolist()
    n_assets = len(tickers)

    mean_returns = (returns_wide.mean() * TRADING_DAYS_PER_YEAR).values
    cov_matrix = (returns_wide.cov() * TRADING_DAYS_PER_YEAR).values

    if objective == "min_vol":
        def neg_objective(w):
            _, vol = _portfolio_stats(w, mean_returns, cov_matrix)
            return vol
    elif objective == "max_sharpe":
        def neg_objective(w):
            ret, vol = _portfolio_stats(w, mean_returns, cov_matrix)
            if vol == 0:
                return np.inf
            return -((ret - risk_free) / vol)
    else:
        raise ValueError(f"optimize_portfolio: unknown objective '{objective}' "
                          f"(expected 'max_sharpe' or 'min_vol')")

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    x0 = np.repeat(1 / n_assets, n_assets)

    result = minimize(neg_objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"optimize_portfolio: optimizer failed to converge ({result.message})")

    weights = dict(zip(tickers, result.x))
    ret, vol = _portfolio_stats(result.x, mean_returns, cov_matrix)
    sharpe = (ret - risk_free) / vol if vol != 0 else np.nan

    return {
        "objective": objective,
        "weights": weights,
        "Annualized_Return": ret,
        "Annualized_Volatility": vol,
        "Sharpe_Ratio": sharpe,
    }
