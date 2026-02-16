import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.pipeline import ensure_dirs  # type: ignore
from analysis.phase_sensor_baselines import eval_models, subset_sensor, col_is_feature  # type: ignore
from analysis.windowing import build_windowed_table  # type: ignore


PHASES = ["pre_uturn", "uturn", "post_uturn", "gait_full"]


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


def choose_balanced_trials(base: Path, per_group: int = None) -> List[str]:
    by_grp = list_trials_by_group(base)
    trials: List[str] = []
    for g in ["healthy", "ortho", "neuro"]:
        arr = by_grp.get(g, [])
        if not arr:
            continue
        k = len(arr) if per_group is None else min(per_group, len(arr))
        trials.extend(arr[:k])
    return trials


def run_sweep(
    base_path: str,
    out_dir: str,
    windows: List[float],
    overlap: float,
    sensors: List[str],  # e.g., ["RF", "ALL"]
    limit_trials: int = None,
) -> Path:
    base = Path(base_path)
    out = Path(out_dir)
    ensure_dirs(out)

    per_group = None
    if limit_trials:
        # approximate per-group budget
        per_group = max(1, limit_trials // 3)
    trials = choose_balanced_trials(base, per_group)
    if not trials:
        raise SystemExit("No trials found under dataset/data (balanced scan)")

    rows: List[Dict[str, object]] = []
    best_per_phase: Dict[str, Tuple[float, float]] = {}  # phase -> (win, bacc)

    for phase in PHASES:
        for win_s in windows:
            # Build (or reuse) window features
            p_csv = out / f"windows_features_{phase}_{int(round(win_s*1000))}ms.csv"
            if not p_csv.exists():
                df = build_windowed_table(base_path, trials, phase, win_s, overlap)
                if df.empty:
                    continue
                df.to_csv(p_csv, index=False)
            else:
                df = pd.read_csv(p_csv)

            if df.empty:
                continue
            # Full-feature space
            feat_cols = [c for c in df.columns if col_is_feature(c)]
            X_all = df[feat_cols].apply(pd.to_numeric, errors="coerce")
            y = df["label"].astype(str)
            groups = df["subject_id"].astype(str)
            # Skip if this phase has <2 classes across all windows
            if len(pd.unique(y)) < 2:
                continue

            # Evaluate for requested sensor settings
            for s in sensors:
                if s == "ALL":
                    X = X_all
                else:
                    # select columns that start with sensor prefix before first '__'
                    cols = [c for c in feat_cols if c.split("__")[0].split("_")[0] == s]
                    if not cols:
                        continue
                    X = X_all[cols]
                model, stats = eval_models(X, y, groups)
                rows.append({
                    "phase": phase,
                    "sensor": s,
                    "win_s": float(win_s),
                    "overlap": float(overlap),
                    "model": model,
                    **stats,
                })
                # Track best per phase using RF-only sensor by BAcc (if RF is part of sensors)
                if s == "RF":
                    bacc = float(stats.get("bacc_mean", np.nan))
                    cur = best_per_phase.get(phase)
                    if cur is None or (not np.isnan(bacc) and bacc > cur[1]):
                        best_per_phase[phase] = (win_s, bacc)

    out_csv = Path(out_dir) / "window_experiments_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Also write a small best-per-phase index for convenience
    best_rows = [
        {"phase": ph, "best_win_s_by_RF": ws, "best_bacc_by_RF": ba}
        for ph, (ws, ba) in sorted(best_per_phase.items())
    ]
    if best_rows:
        pd.DataFrame(best_rows).to_csv(Path(out_dir) / "window_best_per_phase.csv", index=False)

    # Write design notes
    design_md = Path(out_dir) / "window_design.md"
    design_md.write_text(
        "\n".join(
            [
                "# Window Design",
                f"- Windows: {', '.join(str(w) for w in windows)} s",
                f"- Overlap: {overlap*100:.0f}%",
                "- Rationale: 1.0–1.28 s approximate one gait cycle at 100–120 steps/min; 2.56 s provides multi-step context while keeping stationarity. 50% overlap balances temporal resolution and sample independence.",
                "- Grouping: subject-wise 5-fold CV; trials selected in a balanced manner across healthy/ortho/neuro.",
            ]
        ),
        encoding="utf-8",
    )

    return out_csv


def main():
    ap = argparse.ArgumentParser(description="Sliding-window sweep and per-phase evaluation")
    ap.add_argument("--data", default="dataset/data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--windows", default="1.0,1.28,2.56")
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--sensors", default="RF,ALL")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    wins = [float(x) for x in str(args.windows).split(",") if str(x).strip()]
    sens = [s.strip().upper() for s in str(args.sensors).split(",") if s.strip()]
    run_sweep(args.data, args.out, wins, float(args.overlap), sens, args.limit)


if __name__ == "__main__":
    main()
