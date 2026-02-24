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

    # ---- Sequence ----
    "seq_len": 256,                 # 400 not necessary; 256 faster + sufficient
    "target_horizon": 5,

    # ---- Label threshold (COST AWARE) ----
    # 0.002 = 20 bps = realistic after fees/slippage
    "return_threshold": 0.002,

    # ---- Architecture ----
    "num_classes": 3,
    "hidden_size": 256,
    "num_heads": 8,
    "num_layers": 4,                # 6 not needed for this signal
    "dropout": 0.1,
    "ff_mult": 4,
    "head_hidden_mult": 2,

    # ---- Optimization ----
    "batch_size": 1536,             # sweet spot on T4
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "num_epochs": 8,                # don’t overfit noise
    "warmup_steps": 300,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "mixed_precision": "bf16",      # T4 supports bf16 efficiently

    # ---- Multi-scale ----
    "downsample_factors": [1, 5, 15],
    "patch_sizes": [5, 2, 1],

    # ---- Multi-task (off for now) ----
    "use_magnitude_head": False,
    "use_volatility_head": False,
    "magnitude_loss_weight": 0.0,
    "volatility_loss_weight": 0.0,

    # ---- Data split ----
    "purge_gap_samples": 270,
    "train_frac": 0.8,
    "val_frac": 0.1,

    # ---- DataLoader ----
    "num_workers": 2,
    "persistent_workers": False,
    "label_smoothing": 0.0,
}


# =========================
# Paths
# =========================
CHECKPOINT_DIR = "./checkpoints/"
LOG_DIR = "./logs/"
