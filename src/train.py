"""
train.py
--------
Trains the WinProbabilityNet on data/processed/features.csv, evaluates it,
and saves the trained weights + feature scaler to models/.

Usage:
    python src/features.py            # build data/processed/features.csv first
    python src/train.py --epochs 30
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import GroupShuffleSplit

from model import FEATURE_COLUMNS, WinProbabilityNet

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_dataset(path: str = None):
    path = path or os.path.join(PROCESSED_DIR, "features.csv")
    df = pd.read_csv(path)
    return df


def split_by_game(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df["game_id"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def compute_norm_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for col in FEATURE_COLUMNS:
        stats[col] = {"mean": float(df[col].mean()), "std": float(df[col].std() or 1.0)}
    return stats


def normalize(df: pd.DataFrame, stats: dict) -> np.ndarray:
    cols = []
    for col in FEATURE_COLUMNS:
        mean, std = stats[col]["mean"], stats[col]["std"]
        cols.append(((df[col] - mean) / (std if std != 0 else 1.0)).values)
    return np.stack(cols, axis=1).astype(np.float32)


def train(args):
    df = load_dataset()
    train_df, test_df = split_by_game(df, test_size=args.test_size, seed=args.seed)

    norm_stats = compute_norm_stats(train_df)
    X_train = normalize(train_df, norm_stats)
    y_train = train_df["label"].values.astype(np.float32)
    X_test = normalize(test_df, norm_stats)
    y_test = test_df["label"].values.astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WinProbabilityNet(n_features=len(FEATURE_COLUMNS), hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)

    n = len(X_train_t)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                test_probs = torch.sigmoid(model(X_test_t)).cpu().numpy()
            acc = accuracy_score(y_test, (test_probs > 0.5).astype(int))
            ll = log_loss(y_test, np.clip(test_probs, 1e-6, 1 - 1e-6))
            brier = brier_score_loss(y_test, test_probs)
            print(f"epoch {epoch:3d}/{args.epochs}  train_loss={epoch_loss:.4f}  "
                  f"val_acc={acc:.4f}  val_logloss={ll:.4f}  val_brier={brier:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "win_probability_net.pt"))
    with open(os.path.join(MODELS_DIR, "norm_stats.json"), "w") as f:
        json.dump({"feature_columns": FEATURE_COLUMNS, "stats": norm_stats,
                    "hidden": args.hidden}, f, indent=2)

    print(f"[train] Saved model -> {MODELS_DIR}/win_probability_net.pt")
    print(f"[train] Saved norm stats -> {MODELS_DIR}/norm_stats.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
