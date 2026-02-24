# features.py
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def compute_features(df):
    """Add order flow features to the raw dataframe."""
    # Ensure sorted
    df = df.sort_values('timestamp')
    
    # Returns
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility (rolling 20 periods ~20 minutes)
    df['volatility'] = df['return'].rolling(20).std()
    
    # Volume features
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Spread (if we had bid/ask, but we don't; proxy with high-low range)
    df['range'] = (df['high'] - df['low']) / df['close']
    
    # Order flow imbalance proxy: 
    # Using taker buy volume if available (Binance provides taker buy base volume in klines? Yes, column 9)
    # But our simplified klines didn't include it. Let's add it from raw data if we had.
    # For now, we'll use volume * price as a proxy for buying pressure? Not good.
    # Better: We'll keep raw and later incorporate order book data.
    
    # Instead, we'll just compute some technical indicators
    # Simple moving averages
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Drop NaN rows
    df = df.dropna().reset_index(drop=True)
    return df

def process_all_raw():
    """Read all raw parquet files, compute features, and save to processed."""
    raw_path = Path(RAW_DATA_DIR)
    processed_path = Path(PROCESSED_DATA_DIR)
    processed_path.mkdir(parents=True, exist_ok=True)
    
    for month_dir in sorted(raw_path.iterdir()):
        if not month_dir.is_dir():
            continue
        for parquet_file in month_dir.glob("*.parquet"):
            print(f"Processing {parquet_file}...")
            df = pd.read_parquet(parquet_file)
            df_features = compute_features(df)
            
            # Save in similar structure
            out_file = processed_path / month_dir.name / parquet_file.name
            out_file.parent.mkdir(parents=True, exist_ok=True)
            df_features.to_parquet(out_file, index=False)
            print(f"Saved features to {out_file}")

if __name__ == "__main__":
    process_all_raw()