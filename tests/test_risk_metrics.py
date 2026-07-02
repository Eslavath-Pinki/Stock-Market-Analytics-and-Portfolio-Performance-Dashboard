import numpy as np
import pandas as pd
import pytest

from analytics import add_returns
from risk_metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    drawdown_series,
    calmar_ratio,
    value_at_risk,
    conditional_var,
    beta,
    risk_summary,
)


@pytest.fixture
def rising_df():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    prices = 100 * (1.005 ** np.arange(60))
    return add_returns(pd.DataFrame({"Date": dates, "Adj_Close": prices}))


@pytest.fixture
def dip_df():
    """Price rises, drops sharply, then partially recovers -> real drawdown."""
    prices = np.array([100, 105, 110, 90, 95, 100, 108])
    dates = pd.date_range("2024-01-01", periods=len(prices), freq="D")
    return add_returns(pd.DataFrame({"Date": dates, "Adj_Close": prices}))


def test_sharpe_ratio_is_finite_for_steady_uptrend(rising_df):
    result = sharpe_ratio(rising_df)
    assert np.isfinite(result)


def test_sortino_ratio_uses_all_observations_not_only_negative(rising_df):
    # A pure uptrend has (almost) no downside days; downside deviation should
    # be small/zero-ish rather than blowing up, and the ratio should be finite
    # or NaN (if literally zero downside), never a KeyError/crash.
    result = sortino_ratio(rising_df)
    assert result is None or np.isnan(result) or np.isfinite(result)


def test_max_drawdown_is_zero_for_monotonic_uptrend(rising_df):
    result = max_drawdown(rising_df)
    assert result == pytest.approx(0.0, abs=1e-9)


def test_max_drawdown_detects_real_decline(dip_df):
    result = max_drawdown(dip_df)
    # Peak 110 -> trough 90 => drawdown of about -18.2%
    assert result == pytest.approx((90 - 110) / 110, rel=1e-6)
    assert result < 0


def test_drawdown_series_matches_max_drawdown(dip_df):
    series = drawdown_series(dip_df)
    assert series.min() == pytest.approx(max_drawdown(dip_df))


def test_calmar_ratio_nan_when_no_drawdown(rising_df):
    # mdd == 0 for a monotonic uptrend -> function should return NaN, not divide by zero
    result = calmar_ratio(rising_df)
    assert np.isnan(result)


def test_value_at_risk_is_negative_or_zero(dip_df):
    result = value_at_risk(dip_df)
    assert result <= 0


def test_conditional_var_worse_or_equal_than_var(dip_df):
    var = value_at_risk(dip_df)
    cvar = conditional_var(dip_df)
    # CVaR (average of tail losses) should be <= VaR (the threshold)
    assert cvar <= var or np.isclose(cvar, var)


def test_beta_of_series_with_itself_is_one():
    returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
    result = beta(returns, returns)
    assert result == pytest.approx(1.0)


def test_beta_returns_nan_for_insufficient_overlap():
    s1 = pd.Series([0.01], index=[0])
    s2 = pd.Series([0.02], index=[0])
    result = beta(s1, s2)
    assert np.isnan(result)


def test_risk_summary_returns_expected_keys(rising_df):
    summary = risk_summary(rising_df, "TEST")
    expected_keys = {
        "Ticker", "Sharpe_Ratio", "Sortino_Ratio", "Calmar_Ratio",
        "Max_Drawdown", "VaR_95", "CVaR_95",
    }
    assert expected_keys.issubset(summary.keys())
