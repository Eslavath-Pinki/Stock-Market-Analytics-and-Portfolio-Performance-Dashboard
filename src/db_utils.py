"""
MySQL connection and load helpers using SQLAlchemy + PyMySQL.

Usage:
    from db_utils import get_engine, load_companies, load_daily_prices

    engine = get_engine()
    load_companies(engine)
    load_daily_prices(engine, combined_df)

All load_* functions are upserts (INSERT ... ON DUPLICATE KEY UPDATE), so
re-running data_collection.py + this loader after new trading days arrive is
safe: existing (ticker, date) rows are refreshed in place instead of throwing
a duplicate-key error, and re-running the same day twice is a no-op.
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from config import TICKERS, DB_CONFIG

load_dotenv()  # pulls DB credentials from a local .env file if present


def get_engine():
    """Create a SQLAlchemy engine, preferring .env values over config.py defaults."""
    host = os.getenv("DB_HOST", DB_CONFIG["host"])
    port = os.getenv("DB_PORT", DB_CONFIG["port"])
    user = os.getenv("DB_USER", DB_CONFIG["user"])
    password = os.getenv("DB_PASSWORD", DB_CONFIG["password"])
    database = os.getenv("DB_NAME", DB_CONFIG["database"])

    conn_str = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    return create_engine(conn_str)


def _upsert(engine, df: pd.DataFrame, table: str, update_cols: list, chunksize: int = 500) -> None:
    """Generic INSERT ... ON DUPLICATE KEY UPDATE for a DataFrame, in chunks.

    Builds one parameterized multi-row INSERT per chunk (fast, still safe from
    SQL injection since values are bound params, not string-formatted).
    """
    if df.empty:
        return

    cols = list(df.columns)
    col_list = ", ".join(cols)
    placeholders = ", ".join(f":{c}" for c in cols)
    update_clause = ", ".join(f"{c} = VALUES({c})" for c in update_cols)

    stmt = text(f"""
        INSERT INTO {table} ({col_list})
        VALUES ({placeholders})
        ON DUPLICATE KEY UPDATE {update_clause}
    """)

    records = df.to_dict(orient="records")
    with engine.begin() as conn:
        for start in range(0, len(records), chunksize):
            chunk = records[start:start + chunksize]
            conn.execute(stmt, chunk)


def load_companies(engine):
    """Insert/update the companies reference table from config.TICKERS."""
    rows = [{"ticker": t, "company_name": v["name"], "sector": v["sector"], "currency": v["currency"]}
             for t, v in TICKERS.items()]
    df = pd.DataFrame(rows)
    _upsert(engine, df, "companies", update_cols=["company_name", "sector", "currency"])
    print(f"Loaded {len(df)} companies.")


def load_daily_prices(engine, combined_df: pd.DataFrame):
    """Upsert a combined OHLCV DataFrame (from data_collection.py) into daily_prices."""
    df = combined_df.rename(columns={
        "Date": "price_date", "Ticker": "ticker", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Adj_Close": "adj_close", "Volume": "volume",
    })
    df = df[["ticker", "price_date", "open", "high", "low", "close", "adj_close", "volume"]]

    _upsert(engine, df, "daily_prices",
            update_cols=["open", "high", "low", "close", "adj_close", "volume"])
    print(f"Loaded {len(df)} price rows.")


def load_risk_summary(engine, risk_df: pd.DataFrame, as_of_date):
    """Upsert a risk_metrics.risk_summary()-style DataFrame into the risk_summary table.

    schema.sql defines a UNIQUE KEY on (ticker, as_of_date), so re-running the
    analysis for the same day refreshes that day's snapshot in place instead
    of accumulating duplicate rows.
    """
    df = risk_df.copy()
    df["as_of_date"] = as_of_date
    df = df.rename(columns={
        "Ticker": "ticker", "Sharpe_Ratio": "sharpe_ratio", "Sortino_Ratio": "sortino_ratio",
        "Calmar_Ratio": "calmar_ratio", "Max_Drawdown": "max_drawdown",
        "VaR_95": "var_95", "CVaR_95": "cvar_95",
    })
    df = df[["ticker", "as_of_date", "sharpe_ratio", "sortino_ratio",
             "calmar_ratio", "max_drawdown", "var_95", "cvar_95"]]

    _upsert(engine, df, "risk_summary",
            update_cols=["sharpe_ratio", "sortino_ratio", "calmar_ratio",
                         "max_drawdown", "var_95", "cvar_95"])
    print(f"Loaded risk summary for {len(df)} tickers.")


def query(engine, sql: str) -> pd.DataFrame:
    """Run an arbitrary SELECT and return a DataFrame."""
    return pd.read_sql(text(sql), engine)
