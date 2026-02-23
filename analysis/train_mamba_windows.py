import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.datasets.window_raw import WindowedRawIndex, WindowedRawDataset, list_trials  # type: ignore
from analysis.models.mamba_ts import TinyMambaTS  # type: ignore


def seed_all(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str | None = None) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_fold(model: nn.Module, device: torch.device, X_idx: List[int], dataset: WindowedRawDataset,
               y_map: Dict[str, int], epochs: int = 20, batch: int = 32, lr: float = 3e-3) -> None:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        np.random.shuffle(X_idx)
        total = 0.0
        for i in range(0, len(X_idx), batch):
            batch_idx = X_idx[i:i+batch]
            xs = []
            ys = []
            for j in batch_idx:
                x, y, _ = dataset.get(j)
                xs.append(torch.from_numpy(x))
                ys.append(y_map[y])
            xb = torch.stack(xs, dim=0).to(device)
            yb = torch.tensor(ys, dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(batch_idx)
        sched.step()


def eval_fold(model: nn.Module, device: torch.device, X_idx: List[int], dataset: WindowedRawDataset,
              y_map: Dict[str, int]) -> Tuple[float, float]:
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for j in X_idx:
            x, y, _ = dataset.get(j)
            xb = torch.from_numpy(x)[None, ...].to(device)
            logits = model(xb)
            pred = int(torch.argmax(logits, dim=-1).item())
            ys.append(y_map[y])
            ps.append(pred)
    ys = np.array(ys, dtype=int)
    ps = np.array(ps, dtype=int)
    bacc = balanced_accuracy_score(ys, ps)
    f1 = f1_score(ys, ps, average="macro")
    return float(bacc), float(f1)


def run_cv(base: str, phase: str, win: float, overlap: float, device_name: str,
           d_model: int = 64, n_layers: int = 2, epochs: int = 20, batch: int = 32) -> Dict[str, float]:
    base_path = Path(base)
    trials = list_trials(base_path)
    idx = WindowedRawIndex(base, trials, phase, win, overlap)
    ds = WindowedRawDataset(idx)
    labels = np.array(ds.index.labels())
    groups = np.array(ds.index.groups())
    classes = sorted(set(labels.tolist()))
    y_map = {c: i for i, c in enumerate(classes)}
    y_enc = np.array([y_map[y] for y in labels], dtype=int)

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    device = get_device(device_name)
    baccs: List[float] = []
    f1s: List[float] = []

    for tr, te in skf.split(np.zeros_like(y_enc), y_enc, groups):
        # fold-wise normalization using train indices only
        ds.compute_channel_stats(tr.tolist())
        # model per fold (small enough)
        model = TinyMambaTS(in_chans=len(ds.channels), n_classes=len(classes), d_model=d_model, n_layers=n_layers, dropout=0.1)
        model.to(device)
        train_fold(model, device, tr.tolist(), ds, y_map, epochs=epochs, batch=batch)
        bacc, f1 = eval_fold(model, device, te.tolist(), ds, y_map)
        baccs.append(bacc)
        f1s.append(f1)

    return {
        "classes": ",".join(classes),
        "phase": phase,
        "win": float(win),
        "overlap": float(overlap),
        "bacc_mean": float(np.mean(baccs)),
        "bacc_std": float(np.std(baccs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
    }


def main():
    ap = argparse.ArgumentParser(description="Tiny Mamba subject-wise CV on raw RF windows")
    ap.add_argument("--data", default="dataset/data")
    ap.add_argument("--phase", default="gait_full")
    ap.add_argument("--win", type=float, default=3.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="results/artifacts")
    args = ap.parse_args()

    seed_all(42)
    out = run_cv(
        base=args.data,
        phase=args.phase,
        win=args.win,
        overlap=args.overlap,
        device_name=args.device,
        d_model=args.d_model,
        n_layers=args.n_layers,
        epochs=args.epochs,
        batch=args.batch,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"mamba_{args.phase}_{int(round(args.win*1000))}ms_ov{int(round(args.overlap*100))}"
    (out_dir / f"metrics_{tag}.json").write_text(__import__("json").dumps(out, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

