import numpy as np
import pandas as pd
import pytest

from portfolio import (
    wide_returns,
    correlation_matrix,
    portfolio_returns,
    portfolio_performance,
    equal_weight_portfolio,
    random_portfolios,
    optimize_portfolio,
)


@pytest.fixture
def combined_long_df():
    """Long-format price data for two tickers, mimicking data_collection.py output."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    rng = np.random.default_rng(0)

    rows = []
    for ticker, drift in [("AAA", 0.0006), ("BBB", 0.0004)]:
        prices = 100 * np.cumprod(1 + rng.normal(drift, 0.01, size=len(dates)))
        for d, p in zip(dates, prices):
            rows.append({"Date": d, "Ticker": ticker, "Adj_Close": p})
    return pd.DataFrame(rows)


@pytest.fixture
def returns_wide(combined_long_df):
    return wide_returns(combined_long_df)


def test_wide_returns_shape_matches_tickers(returns_wide):
    assert set(returns_wide.columns) == {"AAA", "BBB"}
    assert len(returns_wide) > 0


def test_correlation_matrix_diagonal_is_one(returns_wide):
    corr = correlation_matrix(returns_wide)
    assert corr.loc["AAA", "AAA"] == pytest.approx(1.0)
    assert corr.loc["BBB", "BBB"] == pytest.approx(1.0)


def test_equal_weight_portfolio_sums_to_one():
    weights = equal_weight_portfolio(["AAA", "BBB", "CCC"])
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(w == pytest.approx(1 / 3) for w in weights.values())


def test_portfolio_returns_matches_manual_weighted_sum(returns_wide):
    weights = {"AAA": 0.6, "BBB": 0.4}
    result = portfolio_returns(returns_wide, weights)
    expected = returns_wide["AAA"] * 0.6 + returns_wide["BBB"] * 0.4
    pd.testing.assert_series_equal(result, expected.loc[result.index], check_names=False)


def test_portfolio_performance_has_expected_keys(returns_wide):
    weights = equal_weight_portfolio(["AAA", "BBB"])
    perf = portfolio_performance(returns_wide, weights)
    assert {"Annualized_Return", "Annualized_Volatility", "Sharpe_Ratio"}.issubset(perf.keys())


def test_random_portfolios_weights_sum_to_one(returns_wide):
    sim = random_portfolios(returns_wide, n_portfolios=200, seed=1)
    weight_cols = [c for c in sim.columns if c not in ("Return", "Volatility", "Sharpe")]
    row_sums = sim[weight_cols].sum(axis=1)
    assert row_sums.values == pytest.approx(np.ones(len(sim)), abs=1e-6)


def test_random_portfolios_is_deterministic_with_seed(returns_wide):
    sim1 = random_portfolios(returns_wide, n_portfolios=50, seed=42)
    sim2 = random_portfolios(returns_wide, n_portfolios=50, seed=42)
    pd.testing.assert_frame_equal(sim1, sim2)


def test_optimize_portfolio_weights_sum_to_one_and_long_only(returns_wide):
    result = optimize_portfolio(returns_wide, objective="min_vol")
    weights = result["weights"]
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(-1e-6 <= w <= 1 + 1e-6 for w in weights.values())


def test_optimize_portfolio_invalid_objective_raises(returns_wide):
    with pytest.raises(ValueError):
        optimize_portfolio(returns_wide, objective="not_a_real_objective")
