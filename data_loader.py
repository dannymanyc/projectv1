import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import config


class BTCSequenceDataset(Dataset):
    def __init__(self, sequence_length=256, target_horizon=5):

        self.seq_len = sequence_length
        self.target_horizon = target_horizon
        self.return_threshold = config.MODEL_CONFIG["return_threshold"]

        raw_dir = Path(config.RAW_DATA_DIR)

        # 🔥 Recursively find ALL parquet files (robust to nested folders)
        paths = sorted(raw_dir.rglob("*.parquet"))

        if len(paths) == 0:
            raise RuntimeError(
                f"No parquet files found under {raw_dir.resolve()}.\n"
                "Check GCS copy path or folder structure."
            )

        print(f"Loading {len(paths)} parquet files from {raw_dir.resolve()} ...")

        dfs = [pd.read_parquet(p) for p in paths]

        self.data = (
            pd.concat(dfs, ignore_index=True)
              .sort_values("timestamp")
              .reset_index(drop=True)
        )

        # Cache arrays
        self.close = self.data["close"].to_numpy(dtype=np.float32, copy=True)

        # Basic features
        self.X = self.data[
            ["open", "high", "low", "close", "volume", "number_of_trades"]
        ].to_numpy(dtype=np.float32)

        # Build valid index range
        max_start = len(self.data) - self.seq_len - self.target_horizon
        if max_start <= 0:
            raise ValueError("Dataset too small for given sequence_length + target_horizon")

        self.valid_indices = np.arange(max_start, dtype=np.int64)

        # ---- EXTREMES ONLY FILTER ----
        ends = self.valid_indices + self.seq_len

        p0 = self.close[ends - 1]
        p1 = self.close[ends - 1 + self.target_horizon]

        r = (p1 / p0) - 1.0

        keep = np.abs(r) > self.return_threshold
        self.valid_indices = self.valid_indices[keep]

        print(f"Extremes-only dataset size: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):

        start = self.valid_indices[idx]
        end = start + self.seq_len

        x = self.X[start:end]

        # True future return
        p0 = float(self.close[end - 1])
        p1 = float(self.close[end - 1 + self.target_horizon])
        r = (p1 / p0) - 1.0

        # 2-class label (binary classification)
        y = 1 if r > self.return_threshold else 0

        return x, y
