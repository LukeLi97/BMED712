"""Validate the new frequency-sheet feature CSVs provided by teammate.

Checks:
  - Shape and class balance per phase/window/overlap
  - Missing-value rate per feature
  - Class-wise feature distributions (KDE overlays)
  - Sensor-channel coverage (Acc / FreeAcc / Gyr)
  - 8-class (cohort) distribution

Outputs saved to results/validation/
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal

FREQ_DIR = Path(__file__).resolve().parents[2] / "frequency sheets"
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_COLS = {"subject_id", "trial_id", "window_idx", "label", "cohort",
             "phase", "win_s", "overlap"}

SENSORS = ["HE", "LB", "LF", "RF"]
CHANNELS = ["Acc", "FreeAcc", "Gyr"]
AXES = ["X", "Y", "Z"]
FEAT_TYPES = ["mean", "std", "rms", "dom_freq", "spec_centroid", "spec_power"]

LABEL_COLORS = {"healthy": "#2ECC71", "neuro": "#E74C3C", "ortho": "#3498DB"}
COHORT_COLORS = {
    "HS": "#2ECC71", "RIL": "#C0392B", "PD": "#E74C3C",
    "CVA": "#E67E22", "CIPN": "#8E44AD",
    "KOA": "#2980B9", "HOA": "#1ABC9C", "ACL": "#F39C12",
}


def load_all_csvs() -> pd.DataFrame:
    """Load every CSV from all phase subdirectories."""
    dfs = []
    for phase_dir in sorted(FREQ_DIR.iterdir()):
        if not phase_dir.is_dir():
            continue
        for csv in sorted(phase_dir.glob("*.csv")):
            df = pd.read_csv(csv)
            dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def check_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Return table of rows per phase × win_s × overlap combination."""
    rows = []
    for (phase, win, ov), g in df.groupby(["phase", "win_s", "overlap"]):
        class_counts = g["label"].value_counts().to_dict()
        rows.append({
            "phase": phase, "win_s": win, "overlap_pct": int(ov),
            "n_windows": len(g),
            "n_subjects": g["subject_id"].nunique(),
            "healthy": class_counts.get("healthy", 0),
            "neuro": class_counts.get("neuro", 0),
            "ortho": class_counts.get("ortho", 0),
        })
    return pd.DataFrame(rows).sort_values(["phase", "win_s", "overlap_pct"])


def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value rate per feature column."""
    fcols = feature_cols(df)
    miss = df[fcols].isnull().mean()
    return miss[miss > 0].sort_values(ascending=False).reset_index(
        name="missing_rate").rename(columns={"index": "feature"})


def check_sensor_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Confirm all 216 sensor-channel-axis-feature combinations are present."""
    fcols = set(feature_cols(df))
    rows = []
    for s in SENSORS:
        for ch in CHANNELS:
            for ax in AXES:
                for ft in FEAT_TYPES:
                    name = f"{s}_{ch}_{ax}_{ft}"
                    rows.append({"expected": name, "present": name in fcols})
    return pd.DataFrame(rows)


def plot_class_distribution(df: pd.DataFrame, phase: str = "full_gait",
                             win: float = 5.0, ov: int = 50):
    """KDE plots of top discriminative features for one config."""
    sub = df[(df.phase == phase) & (df.win_s == win) & (df.overlap == ov)]
    if sub.empty:
        return

    fcols = feature_cols(sub)
    # find top 12 by Kruskal-Wallis H
    kw_rows = []
    for c in fcols:
        groups = [sub[sub.label == lbl][c].dropna().values
                  for lbl in ["healthy", "neuro", "ortho"]]
        if all(len(g) > 3 for g in groups):
            try:
                H, p = kruskal(*groups)
                kw_rows.append((c, H, p))
            except Exception:
                pass
    if not kw_rows:
        return
    top_feats = sorted(kw_rows, key=lambda x: -x[1])[:12]

    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    fig.suptitle(
        f"Class Distributions — {phase} | {win}s window | {ov}% overlap",
        fontsize=11, fontweight="bold")
    axes = axes.flatten()

    for i, (feat, H, p) in enumerate(top_feats):
        ax = axes[i]
        for lbl, color in LABEL_COLORS.items():
            vals = sub[sub.label == lbl][feat].dropna()
            if len(vals) < 5:
                continue
            vals.plot.kde(ax=ax, label=lbl, color=color, linewidth=1.5)
        ax.set_title(f"{feat}\nH={H:.1f} p={p:.1e}", fontsize=6.5)
        ax.set_xlabel("")
        ax.legend(fontsize=5)
        ax.tick_params(labelsize=6)

    for j in range(len(top_feats), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / f"class_dist_{phase}_win{int(win*1000)}ms_ov{ov}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_cohort_balance(df: pd.DataFrame):
    """Bar chart of window counts per cohort per phase."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)
    phases = sorted(df.phase.unique())
    for ax, phase in zip(axes, phases):
        sub = df[df.phase == phase]
        counts = sub.groupby("cohort").size().sort_values(ascending=False)
        colors = [COHORT_COLORS.get(c, "#aaaaaa") for c in counts.index]
        counts.plot.bar(ax=ax, color=colors, edgecolor="white", width=0.7)
        ax.set_title(phase.replace("_", " ").title(), fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylabel("Windows" if phase == phases[0] else "")
    fig.suptitle("Window Counts per Cohort and Phase (all windows combined)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "cohort_balance.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_feature_variance_by_class(df: pd.DataFrame, phase: str = "full_gait",
                                    win: float = 5.0, ov: int = 50):
    """Heatmap of mean feature value (z-scored) per cohort."""
    sub = df[(df.phase == phase) & (df.win_s == win) & (df.overlap == ov)]
    if sub.empty:
        return

    fcols = feature_cols(sub)
    # Top 30 by variance across class means
    class_means = sub.groupby("cohort")[fcols].mean()
    variance_across = class_means.var(axis=0).sort_values(ascending=False)
    top30 = variance_across.head(30).index.tolist()

    z = class_means[top30].copy()
    z = (z - z.mean()) / (z.std() + 1e-9)

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(z.values, aspect="auto", cmap="RdYlBu_r", vmin=-2, vmax=2)
    ax.set_yticks(range(len(z.index)))
    ax.set_yticklabels(z.index, fontsize=8)
    ax.set_xticks(range(len(top30)))
    ax.set_xticklabels(top30, rotation=90, fontsize=5.5)
    plt.colorbar(im, ax=ax, label="z-score")
    ax.set_title(
        f"Top 30 Features by Cross-Cohort Variance — {phase} {win}s/{ov}%",
        fontsize=9)
    plt.tight_layout()
    out = OUT_DIR / f"feature_heatmap_{phase}_win{int(win*1000)}ms_ov{ov}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def main():
    print("Loading all frequency-sheet CSVs...")
    df = load_all_csvs()
    print(f"Total windows loaded: {len(df):,}")
    print(f"Phases: {sorted(df.phase.unique())}")
    print(f"Columns: {df.shape[1]} ({len(feature_cols(df))} features + metadata)")

    # 1. Coverage table
    print("\n--- Phase/Window/Overlap coverage ---")
    cov = check_coverage(df)
    print(cov.to_string(index=False))
    cov.to_csv(OUT_DIR / "coverage_table.csv", index=False)

    # 2. Missing values
    print("\n--- Missing values ---")
    miss = check_missing(df)
    if miss.empty:
        print("  No missing values found.")
    else:
        print(miss.head(20).to_string(index=False))
    miss.to_csv(OUT_DIR / "missing_values.csv", index=False)

    # 3. Sensor/channel coverage
    print("\n--- Sensor channel coverage ---")
    sc = check_sensor_coverage(df)
    present = sc.present.sum()
    total = len(sc)
    print(f"  {present}/{total} expected feature columns present")
    missing_cols = sc[~sc.present]
    if not missing_cols.empty:
        print("  Missing columns:")
        print(missing_cols.to_string(index=False))
    sc.to_csv(OUT_DIR / "sensor_coverage.csv", index=False)

    # 4. Plots
    print("\n--- Generating plots ---")
    plot_cohort_balance(df)
    for phase in ["full_gait", "pre_uturn", "post_uturn"]:
        plot_class_distribution(df, phase=phase, win=5.0, ov=50)
        plot_feature_variance_by_class(df, phase=phase, win=5.0, ov=50)

    # 5. Quick stats summary
    print("\n--- 8-class distribution (full_gait, 5s/50%) ---")
    sub = df[(df.phase == "full_gait") & (df.win_s == 5.0) & (df.overlap == 50)]
    print(sub.groupby("cohort").size().to_string())

    # 6. Feature completeness check
    fcols = feature_cols(df)
    expected = len(SENSORS) * len(CHANNELS) * len(AXES) * len(FEAT_TYPES)
    print(f"\nFeature completeness: {len(fcols)}/{expected} expected feature columns")

    print(f"\nAll validation outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
