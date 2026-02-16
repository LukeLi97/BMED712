import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS = Path("results")


def pick_best(df: pd.DataFrame, phase: str, sensor: str = "RF") -> Dict:
    sub = df[(df["phase"] == phase) & (df["sensor"] == sensor)].dropna(subset=["bacc_mean"])  # type: ignore[index]
    if sub.empty:
        return {"bacc": np.nan, "win_s": np.nan, "overlap": np.nan}
    r = sub.loc[sub["bacc_mean"].idxmax()]
    return {"bacc": float(r["bacc_mean"]), "win_s": float(r["win_s"]), "overlap": float(r["overlap"])}


def main():
    tpath = RESULTS / "window_experiments_summary.csv"
    fpath = RESULTS / "window_experiments_freq_summary.csv"
    if not (tpath.exists() and fpath.exists()):
        raise SystemExit("Missing summary CSVs for comparison.")
    dt = pd.read_csv(tpath)
    df = pd.read_csv(fpath)
    phases = ["pre_uturn", "post_uturn", "gait_full", "uturn"]
    rows = []
    for ph in phases:
        t = pick_best(dt, ph, "RF")
        f = pick_best(df, ph, "RF")
        rows.append({
            "phase": ph,
            "time_bacc": t["bacc"],
            "time_win_s": t["win_s"],
            "time_overlap": t["overlap"],
            "timefreq_bacc": f["bacc"],
            "timefreq_win_s": f["win_s"],
            "timefreq_overlap": f["overlap"],
        })
    out_csv = RESULTS / "artifacts" / "phase_time_vs_timefreq_rf.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Plot
    phases2 = [r["phase"] for r in rows]
    t_vals = [r["time_bacc"] for r in rows]
    f_vals = [r["timefreq_bacc"] for r in rows]

    x = np.arange(len(phases2))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(x - w / 2, t_vals, width=w, label="Time-only", alpha=0.9)
    ax.bar(x + w / 2, f_vals, width=w, label="Time+Bands", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", " ") for p in phases2])
    ax.set_ylabel("Balanced Accuracy (best RF)")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("RF: Time vs Time+Frequency (best per phase)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    out_png = RESULTS / "figures" / "phase_time_vs_timefreq_rf.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

