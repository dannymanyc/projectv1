# data_loader.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, SYMBOL, INTERVAL

class BTCSequenceDataset(Dataset):
    def __init__(self, data_dir=PROCESSED_DATA_DIR, sequence_length=400, target_horizon=5):
        """
        Args:
            data_dir: directory with processed parquet files.
            sequence_length: number of past minutes to use as input.
            target_horizon: number of minutes ahead to predict.
        """
        self.sequence_length = sequence_length
        self.target_horizon = target_horizon
        
        raw_dir = Path(RAW_DATA_DIR)

        # Grab all parquet files like: data/raw/YYYY-MM/*.parquet
        paths = sorted(raw_dir.glob("*/*.parquet"))

        # If you want to restrict to BTCUSDT + interval, keep this filter.
        # NOTE: your downloader saved files like BTCUSDT_1m_YYYY-MM.parquet (underscore format).
        paths = [p for p in paths if f"{SYMBOL}_{INTERVAL}_" in p.name]

        if not paths:
            raise FileNotFoundError(
                f"No parquet files found.\n"
                f"RAW_DATA_DIR={raw_dir.resolve()}\n"
                f"Looked for: {raw_dir}/YYYY-MM/*.parquet\n"
                f"Filtered by name containing: {SYMBOL}_{INTERVAL}_\n"
                f"cwd={os.getcwd()}\n"
                f"Example dirs present: {sorted([d.name for d in raw_dir.glob('*') if d.is_dir()])[:5]}"
            )

        print(f"Loading {len(paths)} parquet files from {raw_dir.resolve()} ...")
        dfs = [pd.read_parquet(p) for p in paths]
        self.data = (
            pd.concat(dfs, ignore_index=True)
              .sort_values("timestamp")
              .reset_index(drop=True)
        )
        
        # Define features (exclude timestamp and maybe target)
        self.feature_columns = [col for col in self.data.columns if col not in ['timestamp', 'return', 'log_return']]
        # We'll use 'return' as target
        self.target_column = 'return'
        
        # Precompute indices where sequences are valid
        self.valid_indices = list(range(len(self.data) - sequence_length - target_horizon))
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        end = start + self.sequence_length
        
        # Input features
        x = self.data.loc[start:end-1, self.feature_columns].values.astype(np.float32)
        
        # Target: direction of cumulative return over next target_horizon minutes
        future_returns = self.data.loc[end:end+self.target_horizon-1, self.target_column].values
        cum_return = np.sum(future_returns)
        # Classify: up (1) if cum_return > 0.0001, down (0) if < -0.0001, flat (2) otherwise
        threshold = 0.0001
        if cum_return > threshold:
            y = 0  # up
        elif cum_return < -threshold:
            y = 1  # down
        else:
            y = 2  # flat
        
        return torch.tensor(x), torch.tensor(y, dtype=torch.long)

# Example usage
if __name__ == "__main__":
    dataset = BTCSequenceDataset(sequence_length=400, target_horizon=5)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for x_batch, y_batch in loader:
        print(x_batch.shape, y_batch.shape)
        break
