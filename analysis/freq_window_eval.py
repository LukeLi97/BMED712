import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.windowing import build_windowed_table  # type: ignore
from analysis.phase_sensor_baselines import eval_models, col_is_feature  # type: ignore


def list_trials_by_group(base: Path) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {"healthy": [], "ortho": [], "neuro": []}
    for top in ["healthy", "ortho", "neuro"]:
        top_path = base / top
        if not top_path.exists():
            continue
        for cohort in sorted(p for p in top_path.iterdir() if p.is_dir()):
            for subj in sorted(p for p in cohort.iterdir() if p.is_dir()):
                for tr in sorted(p for p in subj.iterdir() if p.is_dir()):
                    groups.setdefault(top, []).append(tr.name)
    return groups


def choose_trials(base: Path, per_group: int | None) -> List[str]:
    by_grp = list_trials_by_group(base)
    out: List[str] = []
    for g in ["healthy", "ortho", "neuro"]:
        arr = by_grp.get(g, [])
        if not arr:
            continue
        k = len(arr) if per_group is None else min(per_group, len(arr))
        out.extend(arr[:k])
    return out


def run_once(data: str, phase: str, win: float, overlap: float, per_group: int | None, out_csv: Path) -> None:
    base = Path(data)
    trials = choose_trials(base, per_group)
    df = build_windowed_table(str(base), trials, phase, win, overlap, freq_bands=[(0,3),(3,8),(8,15)], filterbank=(16,20.0))
    if df.empty:
        print(f"[warn] empty DF for phase={phase}, win={win}")
        return
    feat_cols = [c for c in df.columns if col_is_feature(c)]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    y = df["label"].astype(str)
    groups = df["subject_id"].astype(str)
    model, stats = eval_models(X, y, groups)
    row = {"phase": phase, "win": float(win), "overlap": float(overlap), "model": model}
    for k in ["acc_mean","acc_std","bacc_mean","bacc_std","macro_f1_mean","macro_f1_std"]:
        row[k] = float(stats.get(k, np.nan))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        df_out = pd.read_csv(out_csv)
        df_out = pd.concat([df_out, pd.DataFrame([row])], ignore_index=True)
    else:
        df_out = pd.DataFrame([row])
    df_out.to_csv(out_csv, index=False)
    print(row)


def main():
    ap = argparse.ArgumentParser(description="Evaluate time+frequency windowing (filterbank/bands)")
    ap.add_argument("--data", default="dataset/data")
    ap.add_argument("--phase", required=True)
    ap.add_argument("--win", type=float, required=True)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--limit-per-group", type=int, default=None)
    ap.add_argument("--out", default="results/window_freq_eval.csv")
    args = ap.parse_args()
    run_once(args.data, args.phase, args.win, args.overlap, args.limit_per_group, Path(args.out))


if __name__ == "__main__":
    main()

