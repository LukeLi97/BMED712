import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Ensure repo root on path to import sibling module
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.phase_sensor_baselines import eval_models, col_is_feature  # type: ignore


PHASES = ["pre_uturn", "uturn", "post_uturn", "gait_full"]
PHASE2CSV = {
    "pre_uturn": Path("results/features_pre_uturn.csv"),
    "uturn": Path("results/features_uturn.csv"),
    "post_uturn": Path("results/features_post_uturn.csv"),
    "gait_full": Path("results/features_gait_full.csv"),
}


def load_single_summary(summary_csv: Path) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(summary_csv)
    out: Dict[str, Dict[str, float]] = {}
    for phase in PHASES:
        sub = df[df["phase"] == phase]
        if sub.empty:
            continue
        row = sub.sort_values("bacc_mean", ascending=False).iloc[0]
        out[phase] = {
            "sensor": str(row["sensor"]),
            "bacc_mean": float(row["bacc_mean"]),
            "bacc_std": float(row.get("bacc_std", np.nan)),
        }
    return out


def eval_all_sensors_per_phase() -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for phase, path in PHASE2CSV.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        feat_cols = [c for c in df.columns if col_is_feature(c)]
        X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
        y = df["label"].astype(str)
        groups = df["subject_id"].astype(str)
        model, stats = eval_models(X, y, groups)
        out[phase] = {
            "model": model,
            "bacc_mean": float(stats.get("bacc_mean", np.nan)),
            "bacc_std": float(stats.get("bacc_std", np.nan)),
        }
    return out


def plot(single: Dict[str, Dict[str, float]], full: Dict[str, Dict[str, float]], out_png: Path, out_csv: Path):
    phases = [p for p in PHASES if p in single and p in full]
    if not phases:
        raise SystemExit("No overlapping phases to plot.")

    # Table for export
    rows = []
    for p in phases:
        rows.append({
            "phase": p,
            "single_sensor": single[p]["sensor"],
            "single_bacc": single[p]["bacc_mean"],
            "single_bacc_std": single[p]["bacc_std"],
            "full_model": full[p]["model"],
            "full_bacc": full[p]["bacc_mean"],
            "full_bacc_std": full[p]["bacc_std"],
        })
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Plot grouped bars
    x = np.arange(len(phases))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    s_means = [single[p]["bacc_mean"] for p in phases]
    s_stds = [single[p]["bacc_std"] for p in phases]
    f_means = [full[p]["bacc_mean"] for p in phases]
    f_stds = [full[p]["bacc_std"] for p in phases]

    ax.bar(x - w / 2, s_means, width=w, yerr=s_stds, label="Single (best sensor)", alpha=0.9, capsize=3)
    ax.bar(x + w / 2, f_means, width=w, yerr=f_stds, label="All sensors", alpha=0.9, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", " ") for p in phases])
    ax.set_ylabel("Balanced Accuracy")
    ax.set_ylim(0.5, 0.95)
    ax.set_title("Phase-wise: Single vs All Sensors (5-fold subject-wise)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    results = Path("results")
    summary_csv = results / "sensor_phase_summary.csv"
    single = load_single_summary(summary_csv)
    full = eval_all_sensors_per_phase()
    out_png = results / "figures" / "phase_single_vs_all.png"
    out_csv = results / "artifacts" / "phase_single_vs_all.csv"
    plot(single, full, out_png, out_csv)


if __name__ == "__main__":
    main()

