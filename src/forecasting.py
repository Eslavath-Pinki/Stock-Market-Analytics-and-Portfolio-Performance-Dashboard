"""
Bonus: simple price forecasting.
Two approaches are provided:
  1. Linear regression on time index (fast baseline trend line)
  2. ARIMA (captures autocorrelation better for short horizons)

These are intentionally simple — good enough for a project demo, not for trading.

Also included: a backtest() helper, because a forecast you can't check against
held-out data is just a guess. The previous version had no way to tell whether
ARIMA(5,1,0) — a fixed, arbitrary order — was any good for a given ticker.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

from validation import require_columns, require_nonempty


def linear_trend_forecast(df: pd.DataFrame, price_col: str = "Adj_Close",
                           horizon: int = 30) -> pd.DataFrame:
    """Fit a simple linear regression on time index and project forward `horizon` days."""
    require_nonempty(df, "linear_trend_forecast")
    require_columns(df, ["Date", price_col], "linear_trend_forecast")

    df = df.sort_values("Date").reset_index(drop=True)
    X = np.arange(len(df)).reshape(-1, 1)
    y = df[price_col].values

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(df), len(df) + horizon).reshape(-1, 1)
    forecast = model.predict(future_X)

    last_date = df["Date"].iloc[-1]
    future_dates = pd.bdate_range(start=last_date, periods=horizon + 1)[1:]

    return pd.DataFrame({"Date": future_dates, "Forecast": forecast})


def select_arima_order(series: pd.Series, candidate_orders=None) -> tuple:
    """Pick an ARIMA(p,d,q) order by lowest AIC over a small candidate grid.

    A single hardcoded order (e.g. (5,1,0)) fits some series well and others
    badly — AIC-based selection over a modest grid is cheap and adapts to
    whatever series it's actually given.
    """
    if candidate_orders is None:
        candidate_orders = [
            (p, d, q)
            for p in (0, 1, 2, 5)
            for d in (0, 1)
            for q in (0, 1, 2)
        ]

    best_order, best_aic = None, np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in candidate_orders:
            try:
                fitted = ARIMA(series, order=order).fit()
                if fitted.aic < best_aic:
                    best_aic, best_order = fitted.aic, order
            except Exception:
                continue  # some (p,d,q) combos fail to converge on a given series — skip them

    if best_order is None:
        raise RuntimeError("select_arima_order: no candidate ARIMA order converged for this series.")
    return best_order


def arima_forecast(df: pd.DataFrame, price_col: str = "Adj_Close",
                    horizon: int = 30, order=None) -> pd.DataFrame:
    """Fit an ARIMA(p,d,q) model on the price series and forecast forward.

    If `order` is not given, it is selected automatically via select_arima_order
    (AIC-minimizing search) rather than defaulting to a fixed, arbitrary order.
    """
    require_nonempty(df, "arima_forecast")
    require_columns(df, ["Date", price_col], "arima_forecast")

    df = df.sort_values("Date").reset_index(drop=True)
    series = df[price_col]

    if order is None:
        order = select_arima_order(series)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(series, order=order)
        fitted = model.fit()
        forecast_result = fitted.get_forecast(steps=horizon)

    last_date = df["Date"].iloc[-1]
    future_dates = pd.bdate_range(start=last_date, periods=horizon + 1)[1:]

    result = pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast_result.predicted_mean.values,
        "Lower_CI": forecast_result.conf_int().iloc[:, 0].values,
        "Upper_CI": forecast_result.conf_int().iloc[:, 1].values,
    })
    result.attrs["order"] = order
    return result


def backtest(df: pd.DataFrame, price_col: str = "Adj_Close", test_size: int = 30,
             arima_order=None) -> dict:
    """Hold out the last `test_size` days, forecast them with both methods
    trained on everything before, and score against the real values.

    Returns RMSE and MAE for each method plus which one won, so a forecast
    can be judged instead of just trusted.
    """
    require_nonempty(df, "backtest")
    require_columns(df, ["Date", price_col], "backtest")

    df = df.sort_values("Date").reset_index(drop=True)
    if len(df) <= test_size + 30:
        raise ValueError(
            f"backtest: need more than {test_size + 30} rows to hold out "
            f"{test_size} test days and still have enough training history."
        )

    train, test = df.iloc[:-test_size].copy(), df.iloc[-test_size:].copy()
    actual = test[price_col].values

    linear_fc = linear_trend_forecast(train, price_col=price_col, horizon=test_size)
    order = arima_order or select_arima_order(train[price_col])
    arima_fc = arima_forecast(train, price_col=price_col, horizon=test_size, order=order)

    def _rmse(pred):
        return float(np.sqrt(np.mean((np.asarray(pred) - actual) ** 2)))

    def _mae(pred):
        return float(np.mean(np.abs(np.asarray(pred) - actual)))

    scores = {
        "test_size": test_size,
        "arima_order": order,
        "Linear_RMSE": _rmse(linear_fc["Forecast"].values),
        "Linear_MAE": _mae(linear_fc["Forecast"].values),
        "ARIMA_RMSE": _rmse(arima_fc["Forecast"].values),
        "ARIMA_MAE": _mae(arima_fc["Forecast"].values),
    }
    scores["Better_Model"] = "ARIMA" if scores["ARIMA_RMSE"] < scores["Linear_RMSE"] else "Linear"
    return scores
