# download.py (Binance public data files via data.binance.vision)
import io
import os
import time
import zipfile
import argparse
from pathlib import Path
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

import pandas as pd

from config import SYMBOL, INTERVAL, START_DATE, END_DATE, RAW_DATA_DIR, MARKET, REQUEST_SLEEP_SECONDS

BASE_URL = "https://data.binance.vision/data"

# Map config MARKET -> path prefix used by Binance public data
MARKET_PREFIX = {
    "spot": "spot",
    "futures_um": "futures/um",   # USDⓈ-M futures
    "futures_cm": "futures/cm",   # COIN-M futures (optional)
}

KLINE_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
]


def _detect_and_parse_timestamp(ts_series: pd.Series) -> pd.Series:
    """
    Binance public data may use milliseconds historically, but some spot data
    (2025+) can be in microseconds. Detect by magnitude.
    """
    # Drop NaN just in case
    s = pd.to_numeric(ts_series, errors="coerce")
    if s.dropna().empty:
        raise ValueError("Timestamp column is empty or invalid")

    sample = int(s.dropna().iloc[0])

    # Rough heuristic:
    # ms unix timestamp ~ 13 digits (e.g., 1700000000000)
    # us unix timestamp ~ 16 digits (e.g., 1700000000000000)
    if sample > 10**14:  # likely microseconds
        return pd.to_datetime(s, unit="us", utc=True)
    else:
        return pd.to_datetime(s, unit="ms", utc=True)


def _monthly_zip_url(symbol: str, interval: str, year: int, month: int, market: str) -> str:
    prefix = MARKET_PREFIX[market]
    filename = f"{symbol}-{interval}-{year:04d}-{month:02d}.zip"
    # Example (spot): data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2025-10.zip
    return f"{BASE_URL}/{prefix}/monthly/klines/{symbol}/{interval}/{filename}"


def _daily_zip_url(symbol: str, interval: str, year: int, month: int, day: int, market: str) -> str:
    prefix = MARKET_PREFIX[market]
    filename = f"{symbol}-{interval}-{year:04d}-{month:02d}-{day:02d}.zip"
    return f"{BASE_URL}/{prefix}/daily/klines/{symbol}/{interval}/{filename}"


def _http_get_bytes(url: str, timeout: int = 30) -> bytes | None:
    """
    Returns response bytes or None for 404/missing files.
    Raises other errors.
    """
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except HTTPError as e:
        if e.code == 404:
            return None
        raise
    except URLError:
        raise


def _read_zipped_kline_csv(zip_bytes: bytes) -> pd.DataFrame:
    """
    Reads the first CSV in a Binance kline ZIP archive into a DataFrame.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise ValueError("ZIP file contained no CSV")
        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(f, header=None)

    # Some files may include a header row in rare cases, but Binance public klines are usually headerless.
    # Normalize to expected 12 cols if possible.
    if df.shape[1] < 12:
        raise ValueError(f"Unexpected kline CSV shape: {df.shape}")

    df = df.iloc[:, :12].copy()
    df.columns = KLINE_COLUMNS

    # Keep only needed columns (same as your script)
    df = df[["timestamp", "open", "high", "low", "close", "volume", "number_of_trades"]]

    # Parse timestamps safely (ms/us)
    df["timestamp"] = _detect_and_parse_timestamp(df["timestamp"])

    # Cast numeric columns
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce").fillna(0).astype(int)

    # Drop bad rows if any
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).reset_index(drop=True)
    return df


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    this_month = datetime(year, month, 1)
    return (next_month - this_month).days


def _download_monthly_or_fallback_daily(symbol: str, interval: str, year: int, month: int, market: str) -> pd.DataFrame | None:
    """
    Try monthly zip first. If unavailable, fallback to stitching daily zips.
    """
    monthly_url = _monthly_zip_url(symbol, interval, year, month, market)
    monthly_bytes = _http_get_bytes(monthly_url)

    if monthly_bytes is not None:
        return _read_zipped_kline_csv(monthly_bytes)

    # Fallback to daily files (useful for the current month before monthly file is published)
    parts = []
    for day in range(1, _days_in_month(year, month) + 1):
        daily_url = _daily_zip_url(symbol, interval, year, month, day, market)
        daily_bytes = _http_get_bytes(daily_url)
        if daily_bytes is None:
            continue
        try:
            parts.append(_read_zipped_kline_csv(daily_bytes))
        except Exception as e:
            print(f"Warning: failed to parse daily {year}-{month:02d}-{day:02d}: {e}")

        time.sleep(REQUEST_SLEEP_SECONDS)

    if not parts:
        return None

    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def download_month(symbol, interval, year, month):
    """Download all klines for a given month from Binance public data, skip if file exists."""
    out_dir = Path(RAW_DATA_DIR) / f"{year:04d}-{month:02d}"
    file_path = out_dir / f"{symbol}_{interval}_{year:04d}-{month:02d}.parquet"

    if file_path.exists():
        print(f"Data for {year}-{month:02d} already exists, skipping.")
        return None

    print(f"Downloading {year}-{month:02d} from Binance public data ({MARKET})...")
    try:
        df = _download_monthly_or_fallback_daily(symbol, interval, year, month, MARKET)
    except Exception as e:
        print(f"Download error for {year}-{month:02d}: {e}")
        return None

    if df is None or df.empty:
        print(f"No data for {year}-{month:02d}")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path, index=False)
    print(f"Saved {len(df)} rows to {file_path}")
    return df


def month_iter(start_date: str, end_date: str):
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(day=1)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(day=1)
    current = start
    while current <= end:
        yield current.year, current.month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)


def download_range(start_date, end_date):
    for year, month in month_iter(start_date, end_date):
        download_month(SYMBOL, INTERVAL, year, month)
        time.sleep(REQUEST_SLEEP_SECONDS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    args = parser.parse_args()

    download_range(args.start, args.end)