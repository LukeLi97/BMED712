import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold

from analysis.deep.mamba_model import MiniMamba  # type: ignore


def to_tensor_windows(df: pd.DataFrame, sensor_prefix: str) -> (torch.Tensor, np.ndarray, np.ndarray):
    feat_cols = [c for c in df.columns if c.startswith(sensor_prefix + "_") and "__" in c]
    # group by window id (use the row as one window; features already aggregated)
    X = df[feat_cols].to_numpy(dtype=np.float32)
    # treat each row as a single-step sequence length 1 with channels=len(feat_cols)
    # for a true sequence model we'd feed raw time series; here we keep it simple
    X = torch.from_numpy(X).unsqueeze(1)  # [N, T=1, C]
    y = pd.factorize(df["label"].astype(str))[0].astype(np.int64)
    groups = df["subject_id"].astype(str).to_numpy()
    return X, y, groups


def train_one_fold(model, Xtr, ytr, Xte, yte, device: torch.device):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss()
    def run_epoch(X, y, train=True):
        bs = 128
        n = X.shape[0]
        idx = np.arange(n)
        if train:
            np.random.shuffle(idx)
        tot, corr = 0.0, 0
        if train:
            model.train()
        else:
            model.eval()
        for i in range(0, n, bs):
            j = idx[i:i+bs]
            xb = X[j].to(device)
            yb = torch.from_numpy(y[j]).to(device)
            with torch.set_grad_enabled(train):
                logits = model(xb)
                loss = crit(logits, yb)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()) * len(j)
            pred = logits.argmax(dim=1)
            corr += int((pred == yb).sum().item())
        return tot / n, corr / n
    # short training
    for _ in range(5):
        run_epoch(Xtr, ytr, True)
    _, acc = run_epoch(Xte, yte, False)
    return acc


def main():
    ap = argparse.ArgumentParser(description="MiniMamba quick CV on window features (sanity check)")
    ap.add_argument("--csv", required=True, help="Path to window features CSV (e.g., results/windows/gait_full/features_win3000ms_ov50.csv)")
    ap.add_argument("--sensor", default="RF")
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    X, y, groups = to_tensor_windows(df, args.sensor)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    accs: List[float] = []
    for tr, te in skf.split(np.zeros_like(y), y, groups):
        model = MiniMamba(in_ch=X.shape[-1], d_model=args.d_model, n_layers=args.layers, n_classes=int(y.max()+1))
        acc = train_one_fold(model, X[tr], y[tr], X[te], y[te], device)
        accs.append(acc)
    print({"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs))})


if __name__ == "__main__":
    main()

