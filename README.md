# Stock Market Analytics and Portfolio Performance Dashboard

Analyze historical stock data for a mix of global and Indian companies to understand
performance, measure risk, compare stocks, build a portfolio, and forecast prices.

## Companies Analyzed

| Ticker | Company | Sector | Currency |
|---|---|---|---|
| AAPL | Apple Inc. | Technology | USD |
| MSFT | Microsoft Corp. | Technology | USD |
| TCS.NS | Tata Consultancy Services | IT Services | INR |
| INFY.NS | Infosys Ltd. | IT Services | INR |
| RELIANCE.NS | Reliance Industries | Conglomerate | INR |

Edit `src/config.py` → `TICKERS` to change this list.

## What's Included

- **Returns**: daily, cumulative, and annualized returns per stock
- **Volatility**: daily and annualized, plus rolling (21-day) volatility
- **Moving averages**: 20/50/200-day SMAs (properly NaN until each window has enough real history)
- **Risk metrics**: Sharpe, Sortino, Calmar ratio, max drawdown, VaR (95%), CVaR (95%), beta
- **Portfolio comparison**: correlation matrix, equal-weight portfolio performance,
  a random-portfolio efficient frontier simulation, and true optimizer solutions
  (min-volatility and max-Sharpe, via `scipy.optimize`, long-only/fully-invested)
- **Bonus — forecasting**: linear trend and ARIMA (auto order selection via AIC),
  plus a backtest that scores both methods against real held-out data

## Project Structure

```
Stock-Market-Analytics/
│
├── data/
│   ├── raw/            # per-ticker CSVs downloaded from Yahoo Finance
│   └── processed/      # combined + summary CSVs used by the notebook & dashboard
├── notebooks/
│   └── 01_stock_analysis.ipynb   # full walkthrough: returns → risk → portfolio → forecast
├── sql/
│   └── schema.sql       # MySQL schema (companies, daily_prices, risk_summary, ...)
├── dashboard/           # Power BI / Tableau file(s) go here
├── images/              # exported charts for the report
├── report/              # write-up / findings
├── src/
│   ├── config.py         # tickers, date range, risk-free rate, DB config
│   ├── validation.py     # shared input-validation helpers (clear errors, not KeyErrors)
│   ├── data_collection.py # yfinance download → CSV (retries + per-ticker failure isolation)
│   ├── analytics.py       # returns, volatility, moving averages
│   ├── risk_metrics.py    # Sharpe, Sortino, Calmar, VaR, CVaR, max drawdown, beta
│   ├── portfolio.py       # correlation, weighted portfolio, efficient frontier, optimizer
│   ├── forecasting.py     # linear trend + ARIMA forecasts, order selection, backtest
│   └── db_utils.py        # MySQL load/query helpers (SQLAlchemy, real upserts)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Collect data

```bash
cd src
python data_collection.py
```

This downloads OHLCV history for every ticker in `config.py` (default: 2020-01-01 to today)
and writes per-ticker CSVs to `data/raw/` plus a combined file to
`data/processed/combined_prices.csv`.

### 2. (Optional) Load into MySQL

```bash
mysql -u root -p < sql/schema.sql
```

Create a `.env` file in the project root (see `.env.example`) with your DB credentials, then:

```python
from src.db_utils import get_engine, load_companies, load_daily_prices
import pandas as pd

engine = get_engine()
load_companies(engine)
combined = pd.read_csv("data/processed/combined_prices.csv", parse_dates=["Date"])
load_daily_prices(engine, combined)
```

### 3. Run the analysis

Open `notebooks/01_stock_analysis.ipynb` and run all cells. It computes returns,
volatility, moving averages, risk metrics, portfolio comparisons, and forecasts —
and exports summary CSVs to `data/processed/` for the dashboard.

### 4. Build the dashboard

Point Power BI / Tableau at the exported CSVs in `data/processed/`
(`return_volatility_summary.csv`, `risk_summary.csv`, `correlation_matrix.csv`)
or connect directly to the `stock_analytics` MySQL database.

## Notes

- Risk-free rate and trading-days assumptions live in `src/config.py`.
- Forecasting models (linear trend, ARIMA) are simple baselines for demonstration,
  not intended for actual trading decisions.
- Indian tickers use the `.NS` (NSE) suffix required by Yahoo Finance.

## Changelog — code quality pass

The first working version had 7 real issues, all fixed here and verified against
a synthetic dataset (full pipeline run + notebook executed end-to-end, zero errors):

1. **`db_utils`**: `load_daily_prices`/`load_risk_summary` used plain `INSERT` and
   would throw a duplicate-key error on any re-run. Replaced with a real
   `INSERT ... ON DUPLICATE KEY UPDATE` upsert; `risk_summary` also got a
   `UNIQUE(ticker, as_of_date)` key so same-day re-runs update in place.
2. **`risk_metrics.sortino_ratio`**: was averaging downside deviation over only
   the negative-return subset, which overstates risk. Fixed to the textbook
   formula — square the shortfall (zero if positive), average over *all*
   observations, then square-root.
3. **`portfolio`**: `scipy` was a listed dependency but never used — the
   "efficient frontier" was random search only. Added `optimize_portfolio()`,
   a real `scipy.optimize.minimize` solver for true min-volatility and
   max-Sharpe portfolios (budget + no-short constraints). Also vectorized
   `random_portfolios()` (was a 5,000-iteration Python loop).
4. **`forecasting`**: ARIMA order was hardcoded to `(5,1,0)` regardless of the
   series, and there was no way to tell if a forecast was any good. Added
   `select_arima_order()` (AIC-based grid search) and `backtest()` (holds out
   the last N days, scores both methods with RMSE/MAE against real values).
5. **`analytics.add_moving_averages`**: used `min_periods=1`, so `SMA_200`
   quietly became "average of 1 day" early in the series. Fixed to
   `min_periods=window` — SMAs are now `NaN` until real history exists.
6. **`data_collection`**: one bad/delisted ticker aborted the whole batch, and
   there was no retry on transient network failures. Added retry with
   exponential backoff per ticker, and the batch now skips (and reports)
   failed tickers instead of losing everything.
7. **Input validation**: added `src/validation.py`; calling most functions with
   a missing column or empty DataFrame now raises a clear `ValueError` instead
   of a cryptic `KeyError` several calls later.

Also added: **Calmar ratio** and a **drawdown chart** (risk_metrics), a
**beta-vs-basket** calculation (no external index in this project, so beta is
relative to the equal-weight basket of the 5 tracked tickers), and an
**`optimal_portfolio_weights.csv`** export for the dashboard alongside the
existing summary tables.
