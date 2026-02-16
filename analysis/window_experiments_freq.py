import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.pipeline import ensure_dirs  # type: ignore
from analysis.phase_sensor_baselines import eval_models, col_is_feature  # type: ignore
from analysis.windowing import build_windowed_table  # type: ignore


PHASES = ["pre_uturn", "uturn", "post_uturn", "gait_full"]


def list_trials(base: Path) -> List[str]:
    trials: List[str] = []
    for top in ["healthy", "ortho", "neuro"]:
        top_path = base / top
        if not top_path.exists():
            continue
        for cohort in sorted(p for p in top_path.iterdir() if p.is_dir()):
            for subj in sorted(p for p in cohort.iterdir() if p.is_dir()):
                for tr in sorted(p for p in subj.iterdir() if p.is_dir()):
                    trials.append(tr.name)
    return trials


def run_sweep_freq(
    base_path: str,
    out_dir: str,
    windows: Sequence[float],
    overlaps: Sequence[float],
    sensors: Sequence[str],
    bands: Sequence[Tuple[float, float]],
    out_subdir: str = "windows_freq",
    phases: Sequence[str] = PHASES,
) -> Path:
    base = Path(base_path)
    out = Path(out_dir)
    ensure_dirs(out)

    trials = list_trials(base)
    if not trials:
        raise SystemExit("No trials found under dataset/data")

    rows: List[Dict[str, object]] = []
    for phase in phases:
        for ov in overlaps:
            for win_s in windows:
                # where to store features for this combination
                win_ms = int(round(win_s * 1000))
                ov_pct = int(round(ov * 100))
                phase_dir = out / out_subdir / phase
                phase_dir.mkdir(parents=True, exist_ok=True)
                p_csv = phase_dir / f"features_win{win_ms}ms_ov{ov_pct}.csv"
                if not p_csv.exists():
                    df = build_windowed_table(base_path, trials, phase, win_s, overlap=ov, freq_bands=bands)
                    if df.empty:
                        continue
                    df.to_csv(p_csv, index=False)
                else:
                    df = pd.read_csv(p_csv)
                if df.empty:
                    continue
                feat_cols = [c for c in df.columns if col_is_feature(c)]
                X_all = df[feat_cols].apply(pd.to_numeric, errors="coerce")
                y = df["label"].astype(str)
                groups = df["subject_id"].astype(str)
                if len(pd.unique(y)) < 2:
                    continue
                for s in sensors:
                    if s == "ALL":
                        X = X_all
                    else:
                        cols = [c for c in feat_cols if c.split("__")[0].split("_")[0] == s]
                        if not cols:
                            continue
                        X = X_all[cols]
                    model, stats = eval_models(X, y, groups)
                    rows.append({
                        "phase": phase,
                        "sensor": s,
                        "win_s": float(win_s),
                        "overlap": float(ov),
                        "model": model,
                        **stats,
                    })

    out_csv = out / f"window_experiments_freq_summary.csv"
    new_df = pd.DataFrame(rows)
    if out_csv.exists():
        try:
            old = pd.read_csv(out_csv)
            merged = pd.concat([old, new_df], ignore_index=True)
            # drop duplicates by phase+sensor+win_s+overlap+model
            merged = merged.drop_duplicates(subset=["phase","sensor","win_s","overlap","model"], keep="last")
            merged.to_csv(out_csv, index=False)
        except Exception:
            new_df.to_csv(out_csv, index=False)
    else:
        new_df.to_csv(out_csv, index=False)
    return out_csv


def main():
    ap = argparse.ArgumentParser(description="Sliding-window + frequency-band features sweep")
    ap.add_argument("--data", default="dataset/data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--windows", default="3.0,4.0,5.0,6.0")
    ap.add_argument("--overlaps", default="0.25,0.50")
    ap.add_argument("--sensors", default="RF,ALL")
    ap.add_argument("--bands", default="0-3,3-8,8-15")
    ap.add_argument("--subdir", default="windows_freq")
    ap.add_argument("--phases", default="pre_uturn,uturn,post_uturn,gait_full")
    args = ap.parse_args()

    wins = [float(x) for x in str(args.windows).split(",") if str(x).strip()]
    ovs = [float(x) for x in str(args.overlaps).split(",") if str(x).strip()]
    sens = [s.strip().upper() for s in str(args.sensors).split(",") if s.strip()]
    # parse bands like "0-3,3-8"
    bands: List[Tuple[float, float]] = []
    for token in str(args.bands).split(","):
        token = token.strip()
        if not token:
            continue
        lo, hi = token.split("-")
        bands.append((float(lo), float(hi)))

    phases = [p.strip() for p in str(args.phases).split(",") if p.strip()]
    run_sweep_freq(args.data, args.out, wins, ovs, sens, bands, out_subdir=args.subdir, phases=phases)


if __name__ == "__main__":
    main()
