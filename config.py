# config.py
SYMBOL = "BTCUSDT"
INTERVAL = "1m"              # e.g. 1m, 5m, 1h
MARKET = "spot"              # "spot" or "futures_um" (USDT-M futures)
START_DATE = "2020-01-01"
END_DATE = "2026-01-31"

# Local paths (use temporary storage)
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "./data/processed/"

# Optional: polite sleep between downloads
REQUEST_SLEEP_SECONDS = 0.25

# GCS bucket (set after creating)
GCS_BUCKET = "your-btc-archive"
GCS_RAW_PREFIX = "btc_1m/raw/"
GCS_FEATURES_PREFIX = "btc_1m/features/"

# Binance API (no key needed for public data, but rate limits apply)
BINANCE_BASE_URL = "https://api.binance.us"

# ========== Model Hyperparameters ==========
MODEL_CONFIG = {
    # Existing
    "seq_len": 400,
    "num_classes": 3,
    "hidden_size": 256,
    "num_heads": 8,
    "num_layers": 6,
    "dropout": 0.1,
    "batch_size": 256,
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "num_epochs": 20,
    "warmup_steps": 500,
    "gradient_accumulation_steps": 2,
    "max_grad_norm": 1.0,
    "mixed_precision": "bf16",   # "none" | "fp16" | "bf16"

    # New architecture knobs
    "downsample_factors": [1, 5, 15],
    "patch_sizes": [5, 2, 1],
    "ff_mult": 4,
    "head_hidden_mult": 2,

    # Multi-task heads (optional)
    "use_magnitude_head": False,
    "use_volatility_head": False,
    "magnitude_loss_weight": 0.0,
    "volatility_loss_weight": 0.0,

    # Split / leakage control
    "target_horizon": 5,
    "purge_gap_samples": 405,  # at least seq_len + target_horizon
    "train_frac": 0.8,
    "val_frac": 0.1,

    # Dataloader
    "num_workers": 4,
    "persistent_workers": True,
    "label_smoothing": 0.0,
}
# Paths
CHECKPOINT_DIR = "./checkpoints/"
LOG_DIR = "./logs/"
