import numpy as np
import pandas as pd
import pytest

from analytics import (
    add_returns,
    cumulative_return,
    annualized_return,
    daily_volatility,
    annualized_volatility,
    add_moving_averages,
    add_rolling_volatility,
    summarize_ticker,
)


@pytest.fixture
def price_df():
    """30 trading days of a steadily rising price series."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    prices = 100 * (1.01 ** np.arange(30))  # +1% per day, compounding
    return pd.DataFrame({"Date": dates, "Adj_Close": prices})


def test_add_returns_creates_expected_columns(price_df):
    out = add_returns(price_df)
    assert "Daily_Return" in out.columns
    assert "Log_Return" in out.columns
    # First row has no prior day -> NaN
    assert pd.isna(out["Daily_Return"].iloc[0])
    # Constant +1% growth -> every subsequent daily return ~= 0.01
    assert out["Daily_Return"].iloc[1:].values == pytest.approx(0.01, abs=1e-9)


def test_add_returns_raises_on_missing_column():
    with pytest.raises(ValueError):
        add_returns(pd.DataFrame({"Date": [1, 2]}))


def test_add_returns_raises_on_empty_df():
    with pytest.raises(ValueError):
        add_returns(pd.DataFrame({"Date": [], "Adj_Close": []}))


def test_cumulative_return_starts_near_one(price_df):
    out = add_returns(price_df)
    cum = cumulative_return(out)
    # First value should be ~1 (NaN return filled to 0)
    assert cum.iloc[0] == pytest.approx(1.0)
    # Should be monotonically increasing for a steadily rising series
    assert (cum.diff().dropna() >= 0).all()


def test_annualized_return_positive_for_uptrend(price_df):
    out = add_returns(price_df)
    result = annualized_return(out)
    assert result > 0


def test_daily_and_annualized_volatility_relationship(price_df):
    out = add_returns(price_df)
    daily_vol = daily_volatility(out)
    ann_vol = annualized_volatility(out)
    assert ann_vol == pytest.approx(daily_vol * np.sqrt(252))


def test_add_moving_averages_nan_until_window_full(price_df):
    out = add_moving_averages(price_df, windows=(5,))
    assert out["SMA_5"].iloc[:4].isna().all()
    assert not pd.isna(out["SMA_5"].iloc[4])


def test_add_rolling_volatility_column_present(price_df):
    out = add_returns(price_df)
    out = add_rolling_volatility(out, window=5)
    assert "Rolling_Vol_5d" in out.columns


def test_summarize_ticker_returns_expected_keys(price_df):
    out = add_returns(price_df)
    summary = summarize_ticker(out, "TEST")
    expected_keys = {
        "Ticker", "Annualized_Return", "Annualized_Volatility",
        "Total_Return", "Best_Day", "Worst_Day",
    }
    assert expected_keys.issubset(summary.keys())
    assert summary["Ticker"] == "TEST"
