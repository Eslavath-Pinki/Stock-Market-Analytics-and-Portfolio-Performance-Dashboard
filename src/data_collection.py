"""
Downloads historical OHLCV data for every ticker in config.TICKERS via yfinance
and saves both per-ticker raw CSVs and a combined long-format CSV.

Run from the project root:
    python src/data_collection.py
"""

import os
import time

import pandas as pd
import yfinance as yf

from config import TICKERS, START_DATE, END_DATE, RAW_DATA_DIR, PROCESSED_DATA_DIR


def download_ticker(ticker: str, max_retries: int = 3, backoff_seconds: float = 2.0) -> pd.DataFrame:
    """Download OHLCV data for a single ticker and return a tidy DataFrame.

    Retries with exponential backoff on failure (Yahoo Finance rate-limits or
    drops connections intermittently) before finally raising, so a single
    transient network hiccup doesn't need a full re-run of the batch.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading {ticker} (attempt {attempt}/{max_retries}) ...")
            df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=False, progress=False)

            if df.empty:
                raise ValueError(f"No data returned for {ticker}. Check the symbol or your connection.")

            # yfinance sometimes returns MultiIndex columns even for a single ticker
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df["Ticker"] = ticker
            df = df.rename(columns={"Adj Close": "Adj_Close"})
            return df[["Date", "Ticker", "Open", "High", "Low", "Close", "Adj_Close", "Volume"]]

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = backoff_seconds * (2 ** (attempt - 1))
                print(f"  {ticker}: attempt {attempt} failed ({e}); retrying in {wait:.0f}s ...")
                time.sleep(wait)

    raise RuntimeError(f"Failed to download {ticker} after {max_retries} attempts: {last_error}")


def download_all(tickers=None, max_retries: int = 3) -> pd.DataFrame:
    """Download all configured tickers, save per-ticker + combined CSVs, return combined df.

    A ticker that fails after all retries is skipped (and reported) rather than
    aborting the whole batch — one bad/delisted symbol shouldn't cost you the
    other four tickers' worth of data.
    """
    tickers = tickers or list(TICKERS.keys())
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    all_frames = []
    failed = []
    for ticker in tickers:
        try:
            df = download_ticker(ticker, max_retries=max_retries)
        except Exception as e:
            print(f"  SKIPPING {ticker}: {e}")
            failed.append(ticker)
            continue
        safe_name = ticker.replace(".", "_")
        df.to_csv(f"{RAW_DATA_DIR}/{safe_name}.csv", index=False)
        all_frames.append(df)

    if not all_frames:
        raise RuntimeError(f"download_all: every ticker failed ({failed}). No data to save.")

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    combined.to_csv(f"{PROCESSED_DATA_DIR}/combined_prices.csv", index=False)

    print(f"\nSaved {len(all_frames)}/{len(tickers)} tickers.")
    if failed:
        print(f"Failed after retries: {failed}")
    print(f"Combined file: {PROCESSED_DATA_DIR}/combined_prices.csv ({len(combined)} rows)")
    return combined


if __name__ == "__main__":
    download_all()
