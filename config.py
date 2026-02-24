# config.py
SYMBOL = "BTCUSDT"
INTERVAL = "1m"              # e.g. 1m, 5m, 1h
MARKET = "spot"              # "spot" or "futures_um" (USDT-M futures)
START_DATE = "2020-01-01"
END_DATE = "2026-01-31"

# Local paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "./data/processed/"

# Optional: polite sleep between downloads
REQUEST_SLEEP_SECONDS = 0.25

# GCS bucket (set after creating)
GCS_BUCKET = "btc-parquet-daniel-20260224"
GCS_RAW_PREFIX = "btc_1m/raw/"
GCS_FEATURES_PREFIX = "btc_1m/features/"

# Binance API (no key needed for public data, but rate limits apply)
BINANCE_BASE_URL = "https://api.binance.us"

# =========================
# Choose a training preset
# =========================
PRESET = "FAST_ITER"   # "FAST_ITER" or "FULLER"

# ========== Model Hyperparameters ==========
# Notes:
# - T4 prefers fp16. bf16 may work but fp16 is usually faster/more stable on T4.
# - seq_len=256 is much cheaper than 400 (attention cost).
# - batch_size is the main knob to use more GPU RAM.
# - num_workers=2 is recommended on Colab (4 can slow/freeze).

FAST_ITER = {
    # Core
    "seq_len": 256,
    "num_classes": 3,
    "hidden_size": 256,
    "num_heads": 8,
    "num_layers": 4,
    "dropout": 0.1,

    # Throughput
    "batch_size": 512,                 # if OOM, drop to 256
    "gradient_accumulation_steps": 1,

    # Optim
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "num_epochs": 20,
    "warmup_steps": 500,
    "max_grad_norm": 1.0,
    "mixed_precision": "fp16",         # T4: fp16 recommended

    # Architecture knobs (keep simple for speed)
    "downsample_factors": [1, 5],      # fewer branches = faster
    "patch_sizes": [5, 2],
    "ff_mult": 4,
    "head_hidden_mult": 2,

    # Multi-task heads (off)
    "use_magnitude_head": False,
    "use_volatility_head": False,
    "magnitude_loss_weight": 0.0,
    "volatility_loss_weight": 0.0,

    # Split / leakage control
    "target_horizon": 5,
    "purge_gap_samples": 270,          # >= seq_len + target_horizon (256+5=261), add buffer
    "train_frac": 0.8,
    "val_frac": 0.1,

    # Dataloader (Colab-safe)
    "num_workers": 2,
    "persistent_workers": False,
    "label_smoothing": 0.0,
}

FULLER = {
    # Core (more capacity)
    "seq_len": 256,
    "num_classes": 3,
    "hidden_size": 512,
    "num_heads": 8,
    "num_layers": 6,
    "dropout": 0.1,

    # Throughput
    "batch_size": 256,                 # if GPU RAM still low, try 384/512
    "gradient_accumulation_steps": 1,

    # Optim
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "num_epochs": 20,
    "warmup_steps": 500,
    "max_grad_norm": 1.0,
    "mixed_precision": "fp16",

    # Architecture knobs (keep all branches if you want)
    "downsample_factors": [1, 5, 15],
    "patch_sizes": [5, 2, 1],
    "ff_mult": 4,
    "head_hidden_mult": 2,

    # Multi-task heads (off)
    "use_magnitude_head": False,
    "use_volatility_head": False,
    "magnitude_loss_weight": 0.0,
    "volatility_loss_weight": 0.0,

    # Split / leakage control
    "target_horizon": 5,
    "purge_gap_samples": 270,          # >= 261, buffer
    "train_frac": 0.8,
    "val_frac": 0.1,

    # Dataloader
    "num_workers": 2,
    "persistent_workers": False,
    "label_smoothing": 0.0,
}

MODEL_CONFIG = FAST_ITER if PRESET == "FAST_ITER" else FULLER

# Paths
CHECKPOINT_DIR = "./checkpoints/"
LOG_DIR = "./logs/"
