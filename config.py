# =========================
# Data / Download Settings
# =========================
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
MARKET = "spot"
START_DATE = "2020-01-01"
END_DATE = "2026-01-31"

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "./data/processed/"

REQUEST_SLEEP_SECONDS = 0.25

GCS_BUCKET = "your-btc-archive"
GCS_RAW_PREFIX = "btc_1m/raw/"
GCS_FEATURES_PREFIX = "btc_1m/features/"

BINANCE_BASE_URL = "https://api.binance.us"


# =========================
# Model / Training Config
# =========================
MODEL_CONFIG = {

    "seq_len": 256,
    "target_horizon": 5,
    "return_threshold": 0.002,

    "num_classes": 2,
    "hidden_size": 256,
    "num_heads": 8,
    "num_layers": 4,
    "dropout": 0.1,

    "batch_size": 1536,
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "num_epochs": 8,
    "warmup_steps": 300,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "mixed_precision": "bf16",

    "downsample_factors": [1, 5, 15],
    "patch_sizes": [5, 2, 1],

    "use_magnitude_head": False,
    "use_volatility_head": False,

    "purge_gap_samples": 270,
    "train_frac": 0.8,
    "val_frac": 0.1,

    "num_workers": 2,
    "persistent_workers": False,
    "label_smoothing": 0.0,
}


# =========================
# Paths
# =========================
CHECKPOINT_DIR = "./checkpoints/"
LOG_DIR = "./logs/"
