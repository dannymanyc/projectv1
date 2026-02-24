# train.py
import os
import sys
import math
import argparse
import random
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Optional logging
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CONFIG, CHECKPOINT_DIR, LOG_DIR  # noqa: F401
from model import PatchMultiScaleTimeSeriesTransformer
from data_loader import BTCSequenceDataset


# ---------------------------
# Reproducibility
# ---------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic can slow training; make configurable if needed
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ---------------------------
# Device / AMP helpers
# ---------------------------
def get_device():
    import torch
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_amp_settings(device: torch.device, mixed_precision: str) -> Tuple[bool, Optional[torch.dtype]]:
    """
    mixed_precision in {"none", "fp16", "bf16"}
    Returns (use_amp, autocast_dtype)
    """
    mp = (mixed_precision or "none").lower()

    if device.type != "cuda":
        return False, None

    if mp == "fp16":
        return True, torch.float16
    if mp == "bf16":
        return True, torch.bfloat16

    return False, None


# ---------------------------
# Data split (chronological + purge gap)
# ---------------------------
def make_purged_time_splits(
    dataset,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    purge_gap: int = 0,
):
    """
    Chronological split with purge/embargo around boundaries to reduce leakage
    from overlapping windows.

    Assumes dataset indices are in strict chronological order.

    raw timeline:
      [ train | val | test ]
    after purge:
      [ train_trimmed | gap | val_trimmed | gap | test_trimmed ]

    purge_gap is in samples (not minutes).
    """
    n = len(dataset)
    if n < 10:
        raise ValueError(f"Dataset too small: {n}")

    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("Invalid split fractions")

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    # Trim regions to avoid overlap leakage near boundaries
    train_stop = max(0, train_end - purge_gap)

    val_start = min(n, train_end + purge_gap)
    val_stop = max(val_start, val_end - purge_gap)

    test_start = min(n, val_end + purge_gap)
    test_stop = n

    train_idx = list(range(0, train_stop))
    val_idx = list(range(val_start, val_stop))
    test_idx = list(range(test_start, test_stop))

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            "Purged split produced an empty split. "
            f"n={n}, train_end={train_end}, val_end={val_end}, purge_gap={purge_gap}. "
            "Reduce purge gap or adjust split fractions."
        )

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
        {
            "n_total": n,
            "train_range": (0, train_stop - 1),
            "val_range": (val_start, val_stop - 1),
            "test_range": (test_start, test_stop - 1),
            "purge_gap": purge_gap,
        },
    )


# ---------------------------
# Batch parsing (supports simple or multitask dataset outputs)
# ---------------------------
def _move_to_device(x, device: torch.device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_move_to_device(v, device) for v in x)
    return x


def parse_batch(batch, device: torch.device):
    """
    Supported formats:
      1) (x, y_cls)
      2) (x, y_cls, aux_dict)
      3) {"x": ..., "y": ..., "magnitude": ..., "volatility": ...}

    Returns:
      x, targets_dict
      targets_dict always has key "y"
      optional keys: "magnitude", "volatility"
    """
    if isinstance(batch, dict):
        x = batch["x"]
        y = batch["y"]
        targets = {"y": y}
        if "magnitude" in batch:
            targets["magnitude"] = batch["magnitude"]
        if "volatility" in batch:
            targets["volatility"] = batch["volatility"]
        return _move_to_device(x, device), _move_to_device(targets, device)

    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            x, y = batch
            targets = {"y": y}
            return _move_to_device(x, device), _move_to_device(targets, device)
        elif len(batch) == 3:
            x, y, aux = batch
            targets = {"y": y}
            if isinstance(aux, dict):
                for k in ("magnitude", "volatility"):
                    if k in aux:
                        targets[k] = aux[k]
            elif torch.is_tensor(aux):
                # If aux is a tensor, assume it's magnitude (customize if needed)
                targets["magnitude"] = aux
            return _move_to_device(x, device), _move_to_device(targets, device)

    raise ValueError(f"Unsupported batch format type={type(batch)}")


# ---------------------------
# Model / loss
# ---------------------------
def get_model(config: Dict[str, Any]):
    return PatchMultiScaleTimeSeriesTransformer(
        num_features=config["num_features"],
        seq_len=config["seq_len"],
        num_classes=config["num_classes"],
        d_model=config.get("hidden_size", 256),
        nhead=config.get("num_heads", 8),
        num_layers=config.get("num_layers", 6),
        dropout=config.get("dropout", 0.1),
        downsample_factors=config.get("downsample_factors", [1, 5, 15]),
        patch_sizes=config.get("patch_sizes", [5, 2, 1]),
        dim_feedforward_mult=config.get("ff_mult", 4),
        head_hidden_mult=config.get("head_hidden_mult", 2),
        use_magnitude_head=config.get("use_magnitude_head", False),
        use_volatility_head=config.get("use_volatility_head", False),
    )


def compute_loss(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], config: Dict[str, Any]):
    """
    Main loss = classification CE
    Optional aux losses:
      magnitude: SmoothL1
      volatility: SmoothL1
    """
    logits = outputs["logits"]
    y = targets["y"].long()

    ce = nn.functional.cross_entropy(
        logits,
        y,
        label_smoothing=float(config.get("label_smoothing", 0.0)),
    )

    total_loss = ce
    loss_dict = {"ce": ce.detach().item()}

    mag_w = float(config.get("magnitude_loss_weight", 0.0))
    if mag_w > 0 and "magnitude" in outputs and "magnitude" in targets:
        mag_target = targets["magnitude"].float().view(-1)
        mag_pred = outputs["magnitude"].view(-1)
        mag_loss = nn.functional.smooth_l1_loss(mag_pred, mag_target)
        total_loss = total_loss + mag_w * mag_loss
        loss_dict["magnitude"] = mag_loss.detach().item()

    vol_w = float(config.get("volatility_loss_weight", 0.0))
    if vol_w > 0 and "volatility" in outputs and "volatility" in targets:
        vol_target = targets["volatility"].float().view(-1)
        vol_pred = outputs["volatility"].view(-1)
        vol_loss = nn.functional.smooth_l1_loss(vol_pred, vol_target)
        total_loss = total_loss + vol_w * vol_loss
        loss_dict["volatility"] = vol_loss.detach().item()

    return total_loss, loss_dict


def get_logits(outputs):
    return outputs["logits"] if isinstance(outputs, dict) else outputs


# ---------------------------
# Train / validate
# ---------------------------
def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    config,
    epoch: int,
):
    model.train()

    total_loss_sum = 0.0
    total_examples = 0
    correct = 0

    accumulation_steps = int(config.get("gradient_accumulation_steps", 1))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))

    use_amp, autocast_dtype = get_amp_settings(device, config.get("mixed_precision", "none"))

    optimizer.zero_grad(set_to_none=True)

    progress_bar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        x, targets = parse_batch(batch, device)
        bs = x.size(0)

        # Forward (AMP)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=use_amp,
        ):
            outputs = model(x)
            loss, loss_parts = compute_loss(outputs, targets, config)

        # Keep true (unscaled/un-normalized) loss for logging
        total_loss_sum += loss.detach().item() * bs
        total_examples += bs

        # Accuracy on classifier head
        logits = get_logits(outputs)
        preds = logits.argmax(dim=-1)
        correct += (preds == targets["y"]).sum().item()

        # Normalize loss for gradient accumulation
        loss_to_backprop = loss / accumulation_steps

        if scaler.is_enabled():
            scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()

        should_step = ((batch_idx + 1) % accumulation_steps == 0)
        is_last_batch = (batch_idx + 1 == len(loader))

        # Important: also step on the final partial accumulation
        if should_step or is_last_batch:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            # OneCycleLR should step AFTER optimizer step (per batch update)
            if scheduler is not None:
                scheduler.step()

        # Progress bar
        current_acc = correct / max(total_examples, 1)
        postfix = {
            "loss": f"{loss.detach().item():.4f}",
            "acc": f"{100 * current_acc:.2f}%",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        }
        # Optional aux loss display if present
        if "magnitude" in loss_parts:
            postfix["mag"] = f"{loss_parts['magnitude']:.4f}"
        if "volatility" in loss_parts:
            postfix["vol"] = f"{loss_parts['volatility']:.4f}"

        progress_bar.set_postfix(postfix)

    avg_loss = total_loss_sum / max(total_examples, 1)
    avg_acc = correct / max(total_examples, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, loader, device, config, desc="Validating"):
    model.eval()

    total_loss_sum = 0.0
    total_examples = 0
    correct = 0

    use_amp, autocast_dtype = get_amp_settings(device, config.get("mixed_precision", "none"))

    progress = tqdm(loader, desc=desc, leave=False)
    for batch in progress:
        x, targets = parse_batch(batch, device)
        bs = x.size(0)

        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=use_amp,
        ):
            outputs = model(x)
            loss, _ = compute_loss(outputs, targets, config)

        logits = get_logits(outputs)
        preds = logits.argmax(dim=-1)

        total_loss_sum += loss.detach().item() * bs
        correct += (preds == targets["y"]).sum().item()
        total_examples += bs

        progress.set_postfix({
            "loss": f"{(total_loss_sum / max(total_examples, 1)):.4f}",
            "acc": f"{100 * (correct / max(total_examples, 1)):.2f}%"
        })

    avg_loss = total_loss_sum / max(total_examples, 1)
    avg_acc = correct / max(total_examples, 1)
    return avg_loss, avg_acc


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    config = MODEL_CONFIG.copy()
    config.setdefault("target_horizon", 5)
    config.setdefault("num_classes", 3)
    config.setdefault("mixed_precision", "none")  # one of: none, fp16, bf16
    config.setdefault("gradient_accumulation_steps", 1)
    config.setdefault("max_grad_norm", 1.0)
    config.setdefault("dropout", 0.1)
    config.setdefault("learning_rate", 1e-4)
    config.setdefault("weight_decay", 1e-2)
    config.setdefault("num_epochs", 10)
    config.setdefault("batch_size", 64)
    config.setdefault("warmup_steps", 100)
    config.setdefault("num_workers", 4)
    config.setdefault("pin_memory", device.type == "cuda")
    config.setdefault("persistent_workers", True)
    config.setdefault("label_smoothing", 0.0)

    # New model defaults (safe if missing in config.py)
    config.setdefault("downsample_factors", [1, 5, 15])
    config.setdefault("patch_sizes", [5, 2, 1])
    config.setdefault("ff_mult", 4)
    config.setdefault("head_hidden_mult", 2)
    config.setdefault("use_magnitude_head", False)
    config.setdefault("use_volatility_head", False)
    config.setdefault("magnitude_loss_weight", 0.0)
    config.setdefault("volatility_loss_weight", 0.0)

    # IMPORTANT: purge gap in samples to reduce leakage from overlapping windows
    # Default = seq_len + target_horizon; increase if your feature engineering uses longer lookbacks.
    config.setdefault(
        "purge_gap_samples",
        int(config["seq_len"]) + int(config.get("target_horizon", 5))
    )

    # Build dataset ONCE (assumes chronological ordering)
    full_dataset = BTCSequenceDataset(
        sequence_length=config["seq_len"],
        target_horizon=config["target_horizon"],
    )

    # Infer feature count from one sample
    sample = full_dataset[0]
    if isinstance(sample, dict):
        sample_x = sample["x"]
    elif isinstance(sample, (list, tuple)):
        sample_x = sample[0]
    else:
        raise ValueError("Unsupported dataset sample format")

    config["num_features"] = int(sample_x.shape[1])
    print(f"Detected num_features: {config['num_features']}")
    print(f"Dataset length: {len(full_dataset)}")

    # Chronological + purged split (NO random_split)
    train_dataset, val_dataset, test_dataset, split_info = make_purged_time_splits(
        full_dataset,
        train_frac=float(config.get("train_frac", 0.8)),
        val_frac=float(config.get("val_frac", 0.1)),
        purge_gap=int(config.get("purge_gap_samples", 0)),
    )

    print("Split info:", split_info)
    print(f"Train/Val/Test sizes: {len(train_dataset)} / {len(val_dataset)} / {len(test_dataset)}")

    num_workers = int(config.get("num_workers", 4))
    pin_memory = bool(config.get("pin_memory", device.type == "cuda"))
    persistent_workers = bool(config.get("persistent_workers", True)) and num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,   # okay within train region
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    # Model
    model = get_model(config).to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        betas=tuple(config.get("betas", (0.9, 0.95))),
    )

    # Scheduler: OneCycleLR (step every optimizer step)
    accumulation_steps = int(config["gradient_accumulation_steps"])
    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = max(1, steps_per_epoch * int(config["num_epochs"]))
    warmup_steps = int(config.get("warmup_steps", 0))
    pct_start = min(max(warmup_steps / total_steps, 0.01), 0.5) if total_steps > 1 else 0.1

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(config["learning_rate"]),
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy="cos",
        cycle_momentum=False,
    )

    # AMP scaler (only needed for fp16)
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda" and str(config.get("mixed_precision", "none")).lower() == "fp16")
    )

    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    # W&B
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("wandb requested but not installed. Continuing without wandb.")
    if use_wandb:
        wandb.init(project="btc-orderflow-upgraded", config=config)
        wandb.watch(model, log="gradients", log_freq=200)

    start_epoch = 0
    best_val_acc = -float("inf")

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and ckpt["scaler"] is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                print("Warning: Could not load scaler state; continuing with fresh scaler.")
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_acc = float(ckpt.get("best_val_acc", best_val_acc))
        print(f"Resumed from epoch {start_epoch} | best_val_acc={best_val_acc:.4f}")

    # Training loop
    for epoch in range(start_epoch, int(config["num_epochs"])):
        train_loss, train_acc = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            config=config,
            epoch=epoch,
        )

        val_loss, val_acc = validate(
            model=model,
            loader=val_loader,
            device=device,
            config=config,
            desc="Validating",
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            })

        # Save best checkpoint by val accuracy (you may later switch to Sharpe on validation backtest)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "best_val_acc": best_val_acc,
                "config": config,
                "split_info": split_info,
            }
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            torch.save(ckpt, ckpt_path)
            print(f"Saved best model -> {ckpt_path} (val_acc={best_val_acc:.4f})")

    # Test best model
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(best_ckpt_path):
        raise FileNotFoundError(f"No best checkpoint found at {best_ckpt_path}")

    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model"])

    test_loss, test_acc = validate(
        model=model,
        loader=test_loader,
        device=device,
        config=config,
        desc="Testing",
    )

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    if use_wandb:
        wandb.log({
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_val_acc": best_val_acc,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
