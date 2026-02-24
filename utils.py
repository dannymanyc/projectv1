# inference.py
import torch
import pandas as pd
import numpy as np
from model import TimeSeriesTransformer
from data_loader import BTCSequenceDataset
from config import MODEL_CONFIG, CHECKPOINT_DIR
import argparse

def load_model(checkpoint_path, config):
    model = TimeSeriesTransformer(
        num_features=config["num_features"],
        seq_len=config["seq_len"],
        num_classes=config["num_classes"],
        d_model=config["hidden_size"],
        nhead=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    )
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()
    return model

def predict(model, df_features, seq_len=400):
    """
    df_features: DataFrame with same features as training, at least seq_len rows.
    Returns predicted class and probabilities.
    """
    # Get last seq_len rows
    last_seq = df_features.iloc[-seq_len:][model.feature_columns].values
    # Convert to tensor
    x = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).cuda()  # (1, seq_len, num_features)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = logits.argmax(dim=-1).item()
    return pred, probs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=f"{CHECKPOINT_DIR}/best_model.pt")
    args = parser.parse_args()

    # Load config (assume we saved it during training)
    config = MODEL_CONFIG  # In practice, load from a saved config file

    # Determine num_features (would need to load from saved config or re-infer)
    # For demo, we hardcode or load from a saved config.json.
    # Let's assume we saved config.json in checkpoint dir.
    import json
    with open(f"{CHECKPOINT_DIR}/config.json", 'r') as f:
        saved_config = json.load(f)
    config.update(saved_config)

    model = load_model(args.checkpoint, config)

    # Simulate new data (in practice, fetch from API)
    # For demo, we'll just use the last part of the dataset
    dataset = BTCSequenceDataset(sequence_length=config["seq_len"], target_horizon=5)
    # Get the last sample's features
    # This is just illustrative
    sample_x, _ = dataset[-1]
    df_dummy = pd.DataFrame(sample_x.numpy(), columns=[f"f{i}" for i in range(sample_x.shape[1])])
    pred, probs = predict(model, df_dummy)
    classes = ["UP", "DOWN", "FLAT"]
    print(f"Prediction: {classes[pred]}, Probabilities: UP={probs[0]:.2f}, DOWN={probs[1]:.2f}, FLAT={probs[2]:.2f}")