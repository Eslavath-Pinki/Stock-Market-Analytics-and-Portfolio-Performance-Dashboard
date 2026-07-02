"""
Central configuration for the Stock Market Analytics project.
Edit TICKERS to change which companies are analyzed everywhere in the pipeline.
"""

# Yahoo Finance tickers. Indian stocks use the .NS (NSE) suffix.
TICKERS = {
    "AAPL": {"name": "Apple Inc.", "sector": "Technology", "currency": "USD"},
    "MSFT": {"name": "Microsoft Corp.", "sector": "Technology", "currency": "USD"},
    "TCS.NS": {"name": "Tata Consultancy Services", "sector": "IT Services", "currency": "INR"},
    "INFY.NS": {"name": "Infosys Ltd.", "sector": "IT Services", "currency": "INR"},
    "RELIANCE.NS": {"name": "Reliance Industries", "sector": "Conglomerate", "currency": "INR"},
}

# Historical data window
START_DATE = "2020-01-01"
END_DATE = None  # None = up to today

# Risk-free rate used in Sharpe ratio (annualized, as a decimal). ~India/US short-term proxy.
RISK_FREE_RATE = 0.06

# Trading days per year (used to annualize daily stats)
TRADING_DAYS_PER_YEAR = 252

# File paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

# MySQL connection (override via .env in practice; these are local defaults)
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "your_password_here",
    "database": "stock_analytics",
}
