-- Stock Market Analytics: MySQL schema
-- Run this once against your MySQL server to create the database structure.

CREATE DATABASE IF NOT EXISTS stock_analytics
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE stock_analytics;

-- Reference table of companies being tracked
CREATE TABLE IF NOT EXISTS companies (
    ticker      VARCHAR(20) PRIMARY KEY,
    company_name VARCHAR(100) NOT NULL,
    sector      VARCHAR(50),
    currency    VARCHAR(10)
);

-- Daily OHLCV price history
CREATE TABLE IF NOT EXISTS daily_prices (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    ticker      VARCHAR(20) NOT NULL,
    price_date  DATE NOT NULL,
    open        DECIMAL(16, 4),
    high        DECIMAL(16, 4),
    low         DECIMAL(16, 4),
    close       DECIMAL(16, 4),
    adj_close   DECIMAL(16, 4),
    volume      BIGINT,
    UNIQUE KEY uq_ticker_date (ticker, price_date),
    CONSTRAINT fk_daily_prices_ticker
        FOREIGN KEY (ticker) REFERENCES companies(ticker)
        ON DELETE CASCADE,
    INDEX idx_ticker_date (ticker, price_date)
);

-- Precomputed daily/rolling metrics (populated from Python, optional cache table)
CREATE TABLE IF NOT EXISTS daily_metrics (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    ticker          VARCHAR(20) NOT NULL,
    price_date      DATE NOT NULL,
    daily_return    DECIMAL(10, 6),
    log_return      DECIMAL(10, 6),
    sma_20          DECIMAL(16, 4),
    sma_50          DECIMAL(16, 4),
    sma_200         DECIMAL(16, 4),
    rolling_vol_21d DECIMAL(10, 6),
    UNIQUE KEY uq_metrics_ticker_date (ticker, price_date),
    CONSTRAINT fk_daily_metrics_ticker
        FOREIGN KEY (ticker) REFERENCES companies(ticker)
        ON DELETE CASCADE
);

-- Summary risk metrics, one row per ticker per analysis run
CREATE TABLE IF NOT EXISTS risk_summary (
    id                  BIGINT AUTO_INCREMENT PRIMARY KEY,
    ticker              VARCHAR(20) NOT NULL,
    as_of_date          DATE NOT NULL,
    annualized_return   DECIMAL(10, 6),
    annualized_volatility DECIMAL(10, 6),
    sharpe_ratio        DECIMAL(10, 6),
    sortino_ratio       DECIMAL(10, 6),
    calmar_ratio        DECIMAL(10, 6),
    max_drawdown        DECIMAL(10, 6),
    var_95              DECIMAL(10, 6),
    cvar_95             DECIMAL(10, 6),
    CONSTRAINT fk_risk_summary_ticker
        FOREIGN KEY (ticker) REFERENCES companies(ticker)
        ON DELETE CASCADE,
    -- One snapshot per ticker per day; re-running the same day updates it
    -- in place instead of piling up duplicate rows.
    UNIQUE KEY uq_risk_summary_ticker_date (ticker, as_of_date)
);

-- Saved portfolio definitions for comparison / dashboard use
CREATE TABLE IF NOT EXISTS portfolio_weights (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    portfolio_name  VARCHAR(100) NOT NULL,
    ticker          VARCHAR(20) NOT NULL,
    weight          DECIMAL(6, 4) NOT NULL,
    CONSTRAINT fk_portfolio_ticker
        FOREIGN KEY (ticker) REFERENCES companies(ticker)
        ON DELETE CASCADE
);

-- Convenience view: each ticker's most recent risk snapshot, joined with
-- company info. Powers a single-query "latest risk dashboard" table without
-- the dashboard tool needing to know how to find the max(as_of_date) itself.
CREATE OR REPLACE VIEW latest_risk_summary AS
SELECT c.ticker, c.company_name, c.sector, c.currency,
       r.as_of_date, r.annualized_return, r.annualized_volatility,
       r.sharpe_ratio, r.sortino_ratio, r.calmar_ratio,
       r.max_drawdown, r.var_95, r.cvar_95
FROM companies c
JOIN risk_summary r ON r.ticker = c.ticker
JOIN (
    SELECT ticker, MAX(as_of_date) AS max_date
    FROM risk_summary
    GROUP BY ticker
) latest ON latest.ticker = r.ticker AND latest.max_date = r.as_of_date;
