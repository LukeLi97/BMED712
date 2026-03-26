"""Gait variability & clinical correlation analysis.

Two analyses:
  A. Step-level variability  — within-trial SD/IQR of step/stride times from
     asymmetry_per_step.csv (58 k rows).  Compares H/O/N on gait rhythm
     consistency beyond trial-level CV already computed.
  B. VGA severity correlation — Spearman r between asymmetry metrics and the
     clinician Visual Gait Assessment score (0–4 scale).

Outputs figures step13_*.png and artifact vga_correlation.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.pipeline import ensure_dirs, find_trials  # type: ignore
from dataset.quick_start.load_data import load_trial  # type: ignore

GROUP_ORDER = ["healthy", "ortho", "neuro"]
GROUP_COLORS = {"healthy": "#4CAF50", "ortho": "#FF9800", "neuro": "#F44336"}


# ══════════════════════════════════════════════════════════════════════
# A. Step-level variability
# ══════════════════════════════════════════════════════════════════════


def compute_step_variability(step_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-step data into per-trial variability metrics."""
    rows = []
    for (trial, subject, group), grp in step_df.groupby(["trial", "subject", "group"]):
        stride = grp[grp["metric"] == "stride_time"]["value_s"]
        step   = grp[grp["metric"] == "step_time"]["value_s"]

        def safe_stats(arr):
            arr = arr.dropna()
            if len(arr) < 3:
                return dict(mean=np.nan, sd=np.nan, iqr=np.nan, cv=np.nan)
            m = float(np.mean(arr))
            sd = float(np.std(arr, ddof=1))
            return dict(
                mean=m,
                sd=sd,
                iqr=float(np.percentile(arr, 75) - np.percentile(arr, 25)),
                cv=sd / m if m > 0 else np.nan,
            )

        ss = safe_stats(stride)
        st = safe_stats(step)
        rows.append({
            "trial": trial, "subject": subject, "group": group,
            "stride_mean_s": ss["mean"], "stride_sd_s": ss["sd"],
            "stride_iqr_s": ss["iqr"], "stride_cv": ss["cv"],
            "step_mean_s":  st["mean"], "step_sd_s":  st["sd"],
            "step_iqr_s":   st["iqr"], "step_cv":    st["cv"],
            "n_strides": int((grp["metric"] == "stride_time").sum()),
            "n_steps":   int((grp["metric"] == "step_time").sum()),
        })
    return pd.DataFrame(rows)


def plot_variability_boxplots(df_subj: pd.DataFrame, out_dir: Path):
    """4-panel boxplots: stride_sd, step_sd, stride_cv, step_cv across H/O/N."""
    metrics = [
        ("stride_sd_s",  "Stride SD (s)",    "Within-trial stride time variability"),
        ("step_sd_s",    "Step SD (s)",      "Within-trial step time variability"),
        ("stride_cv",    "Stride CV",        "Stride coefficient of variation"),
        ("step_cv",      "Step CV",          "Step coefficient of variation"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Gait Rhythm Variability by Group (subject-level means)", fontsize=13, fontweight="bold")

    for ax, (col, ylabel, title) in zip(axes, metrics):
        data = [df_subj.loc[df_subj["group"] == g, col].dropna().values for g in GROUP_ORDER]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=2))
        for patch, g in zip(bp["boxes"], GROUP_ORDER):
            patch.set_facecolor(GROUP_COLORS[g])
            patch.set_alpha(0.75)
        ax.set_xticklabels(["Healthy", "Ortho", "Neuro"], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9)

        # KW p-value annotation
        arrs = [d for d in data if len(d) >= 3]
        if len(arrs) >= 2:
            kw_p = stats.kruskal(*arrs).pvalue
            stars = "***" if kw_p < 0.001 else ("**" if kw_p < 0.01 else ("*" if kw_p < 0.05 else "ns"))
            ax.set_xlabel(f"KW {stars}  p={kw_p:.3f}", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "step13_variability_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_variability_heatmap(df_subj: pd.DataFrame, out_dir: Path):
    """Heatmap: group × variability metric showing median and KW significance."""
    metrics = ["stride_sd_s", "step_sd_s", "stride_cv", "step_cv",
               "stride_iqr_s", "step_iqr_s"]
    groups = GROUP_ORDER

    medians = pd.DataFrame(
        {g: [df_subj.loc[df_subj["group"] == g, m].median() for m in metrics] for g in groups},
        index=metrics
    )
    # Normalise each row to [0,1] for colour
    norm = medians.sub(medians.min(axis=1), axis=0).div(
        medians.max(axis=1) - medians.min(axis=1) + 1e-12, axis=0
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(norm.values.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Normalised median (row-wise)")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g.capitalize() for g in groups])
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.replace("_", " ") for m in metrics], rotation=30, ha="right", fontsize=9)
    ax.set_title("Step-level Variability Heatmap (H / O / N)", fontweight="bold")

    for i, metric in enumerate(metrics):
        for j, g in enumerate(groups):
            val = medians.loc[metric, g]
            ax.text(i, j, f"{val:.3f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "step13_variability_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# B. VGA clinical severity correlation
# ══════════════════════════════════════════════════════════════════════


def extract_vga(base_path: str) -> pd.DataFrame:
    """Walk dataset and extract VGA score + asymmetry metadata per trial."""
    rows = []
    for trial_name in find_trials(Path(base_path)):
        try:
            trial = load_trial(base_path, trial_name)
        except Exception:
            continue
        md = trial.get("metadata", {})
        vga = md.get("visualGaitAssessment")
        if vga is None:
            continue
        try:
            vga = float(vga)
        except (TypeError, ValueError):
            continue
        rows.append({
            "trial": trial_name,
            "subject": str(md.get("subject", "")),
            "group": str(md.get("group", "unknown")),
            "pathology_key": str(md.get("pathologyKey", "")),
            "vga": vga,
        })
    return pd.DataFrame(rows)


def merge_vga_asymmetry(vga_df: pd.DataFrame, asym_path: Path) -> pd.DataFrame:
    """Merge VGA scores with asymmetry features."""
    if not asym_path.exists():
        return pd.DataFrame()
    asym = pd.read_csv(asym_path)
    merged = vga_df.merge(asym, on=["trial", "subject", "group"], how="inner")
    return merged


def plot_vga_scatter(merged: pd.DataFrame, metric: str, out_dir: Path):
    """Scatter plot: VGA score vs asymmetry metric, coloured by group."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for g in GROUP_ORDER:
        sub = merged[merged["group"] == g].dropna(subset=["vga", metric])
        ax.scatter(sub["vga"], sub[metric], label=g.capitalize(),
                   color=GROUP_COLORS[g], alpha=0.6, s=40, edgecolors="none")

    # Overall Spearman
    valid = merged.dropna(subset=["vga", metric])
    if len(valid) >= 5:
        r, p = stats.spearmanr(valid["vga"], valid[metric])
        # Trend line
        z = np.polyfit(valid["vga"], valid[metric], 1)
        xr = np.linspace(valid["vga"].min(), valid["vga"].max(), 100)
        ax.plot(xr, np.polyval(z, xr), "k--", linewidth=1.5,
                label=f"Overall r={r:.2f}, p={p:.3f}")

    ax.set_xlabel("Visual Gait Assessment (0=normal, 4=severe)", fontsize=10)
    ax.set_ylabel(metric.replace("_", " "), fontsize=10)
    ax.set_title(f"VGA vs {metric.replace('_', ' ')}", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / f"step13_vga_{metric}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_vga_summary(corr_df: pd.DataFrame, out_dir: Path):
    """Horizontal bar chart of Spearman r for each metric with VGA."""
    corr_df = corr_df.sort_values("spearman_r", key=abs, ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(4, len(corr_df) * 0.45)))
    colors = ["#d32f2f" if r > 0 else "#1976d2" for r in corr_df["spearman_r"]]
    bars = ax.barh(corr_df["metric"].str.replace("_", " "), corr_df["spearman_r"],
                   color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman r (vs VGA)", fontsize=10)
    ax.set_title("Correlation of Asymmetry Metrics with Clinical Severity (VGA)", fontweight="bold")

    for bar, row in zip(bars, corr_df.itertuples()):
        stars = "***" if row.p_value < 0.001 else ("**" if row.p_value < 0.01 else ("*" if row.p_value < 0.05 else "ns"))
        ax.text(bar.get_width() + 0.01 * np.sign(bar.get_width() + 0.001), bar.get_y() + bar.get_height() / 2,
                stars, va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "step13_vga_summary.png", dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════


def run(base_path: str, out_dir: str):
    out = Path(out_dir)
    ensure_dirs(out)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    art_dir = out / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    # ── A. Step-level variability ─────────────────────────────────────
    step_csv = art_dir / "asymmetry_per_step.csv"
    if step_csv.exists():
        print("[gait_variability] Loading asymmetry_per_step.csv …")
        step_df = pd.read_csv(step_csv)
        var_df = compute_step_variability(step_df)
        var_df.to_csv(art_dir / "step_variability.csv", index=False)

        # Subject-level means
        subj_var = (
            var_df.groupby(["subject", "group"]).mean(numeric_only=True).reset_index()
        )
        plot_variability_boxplots(subj_var, fig_dir)
        plot_variability_heatmap(subj_var, fig_dir)

        print("[gait_variability] Variability stats (subject-level medians):")
        for g in GROUP_ORDER:
            sub = subj_var[subj_var["group"] == g]
            print(f"  {g:8s}: stride_sd={sub['stride_sd_s'].median():.4f}s  "
                  f"step_sd={sub['step_sd_s'].median():.4f}s  "
                  f"stride_cv={sub['stride_cv'].median():.4f}")
    else:
        print(f"[gait_variability] {step_csv} not found — skipping variability analysis.")

    # ── B. VGA correlation ───────────────────────────────────────────
    print("[gait_variability] Extracting VGA scores from metadata …")
    vga_df = extract_vga(base_path)
    print(f"[gait_variability] Found {len(vga_df)} trials with VGA scores.")

    if vga_df.empty:
        print("[gait_variability] No VGA data found — skipping correlation analysis.")
        return

    asym_path = art_dir / "asymmetry_per_trial.csv"
    merged = merge_vga_asymmetry(vga_df, asym_path)
    if merged.empty:
        print("[gait_variability] Could not merge VGA with asymmetry — check trial IDs.")
        return

    merged.to_csv(art_dir / "vga_asymmetry_merged.csv", index=False)

    ASYM_METRICS = ["stride_absAI", "step_absAI", "stride_abs_diff",
                    "step_abs_diff", "stride_CV_l", "step_CV_l"]

    corr_rows = []
    for metric in ASYM_METRICS:
        if metric not in merged.columns:
            continue
        valid = merged.dropna(subset=["vga", metric])
        if len(valid) < 5:
            continue
        r, p = stats.spearmanr(valid["vga"], valid[metric])
        corr_rows.append({"metric": metric, "spearman_r": r, "p_value": p, "n": len(valid)})
        if abs(r) >= 0.15:
            plot_vga_scatter(merged, metric, fig_dir)

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(art_dir / "vga_correlation.csv", index=False)

    print("\n[gait_variability] === VGA Correlation Results ===")
    print(corr_df.to_string(index=False))

    if not corr_df.empty:
        plot_vga_summary(corr_df, fig_dir)

    print(f"\n[gait_variability] Figures saved → {fig_dir}/step13_*.png")


if __name__ == "__main__":
    base = str(REPO_ROOT / "dataset" / "data")
    out = str(REPO_ROOT / "results")
    run(base, out)
