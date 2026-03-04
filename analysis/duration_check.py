"""Duration confound check via OLS residualization.

Trial duration_s is the #1 feature (p=1.7e-33), with neuro trials ~3×
longer than healthy. This script tests whether the other significant
features are genuinely capturing gait pathology OR just reflecting the
fact that longer trials produce different aggregate statistics.

Method (Option A — Residualization):
  For each feature x, fit: x ~ duration_s (OLS)
  Replace x with the residual: x_resid = x - predicted(x|duration)
  Re-run Kruskal-Wallis on the residuals.
  Features that survive residualization reflect genuine gait differences
  independent of walking speed / trial length.

Usage:
    python analysis/duration_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.pipeline import ensure_dirs, find_trials
from analysis.train_baseline import collect_features
from analysis.directionality import (
    build_subject_df,
    feature_stats,
    consistency_score,
    classify_direction,
    plot_boxplot_grid,
    plot_single_boxplot,
    GROUPS,
    GROUP_COLORS,
    ASYM_COLS,
)

COVARIATE = "duration_s"


# ── Residualization ──────────────────────────────────────────────────


def residualize(
    df_subj: pd.DataFrame,
    feature_cols: List[str],
    covariate: str = COVARIATE,
) -> pd.DataFrame:
    """Regress `covariate` out of every feature using OLS.

    Returns a copy of df_subj with feature columns replaced by residuals.
    The covariate column itself is kept unchanged for reference.
    """
    if covariate not in df_subj.columns:
        raise ValueError(f"Covariate '{covariate}' not in DataFrame.")

    df_out = df_subj.copy()
    cov_vals = df_subj[covariate].values.astype(float)

    # Design matrix: [1, duration_s]
    X_cov = np.column_stack([np.ones(len(cov_vals)), cov_vals])

    for feat in feature_cols:
        if feat == covariate:
            continue  # Don't residualize the covariate itself
        y_vals = df_subj[feat].values.astype(float)
        mask = np.isfinite(y_vals) & np.isfinite(cov_vals)
        if mask.sum() < 5:
            continue  # Not enough data
        # OLS: beta = (X'X)^-1 X'y
        try:
            beta, _, _, _ = np.linalg.lstsq(X_cov[mask], y_vals[mask], rcond=None)
            predicted = X_cov @ beta
            df_out[feat] = y_vals - predicted
        except Exception:
            pass  # Keep original if OLS fails

    return df_out


# ── Comparison plot ──────────────────────────────────────────────────


def plot_pvalue_scatter(
    stats_orig: pd.DataFrame,
    stats_resid: pd.DataFrame,
    path: Path,
    label_top_n: int = 15,
) -> None:
    """Scatter: -log10(p) original vs -log10(p) after residualization.

    Points above the diagonal y=x retain/gain significance.
    Points below the diagonal lose significance (duration-confounded).
    """
    merged = stats_orig[["feature", "kw_p", "direction"]].merge(
        stats_resid[["feature", "kw_p"]].rename(columns={"kw_p": "kw_p_resid"}),
        on="feature",
    )
    merged = merged.dropna(subset=["kw_p", "kw_p_resid"])

    # Clip p-values to avoid log(0)
    eps = 1e-40
    x = -np.log10(np.clip(merged["kw_p"].values, eps, 1.0))
    y = -np.log10(np.clip(merged["kw_p_resid"].values, eps, 1.0))

    # Colour by whether feature survives (p_resid < 0.05)
    sig_thresh = -np.log10(0.05)
    colors = np.where(y >= sig_thresh, "#4C72B0", "#C44E52")
    sizes = np.where(merged["feature"].isin(ASYM_COLS), 80, 30)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(x, y, c=colors, s=sizes, alpha=0.7, edgecolors="none")

    # Diagonal y=x  (no change in p)
    lim = max(x.max(), y.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.5, label="No change")

    # Significance threshold lines
    ax.axhline(sig_thresh, color="#C44E52", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.axvline(sig_thresh, color="#C44E52", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.fill_betweenx([sig_thresh, lim * 1.1], sig_thresh, lim * 1.1,
                     alpha=0.04, color="#4C72B0", label="Both significant")
    ax.fill_betweenx([0, sig_thresh], sig_thresh, lim * 1.1,
                     alpha=0.04, color="#C44E52", label="Loses significance")

    # Label top features by original p-value
    top_idx = np.argsort(-x)[:label_top_n]
    for i in top_idx:
        feat_name = str(merged["feature"].iloc[i])
        short = feat_name.split("__")[0][:18] + ("…" if len(feat_name) > 18 else "")
        ax.annotate(
            short, (x[i], y[i]),
            xytext=(4, 2), textcoords="offset points",
            fontsize=6.5, alpha=0.8,
        )

    ax.set_xlabel("-log₁₀(p) Original", fontsize=12)
    ax.set_ylabel("-log₁₀(p) After Removing duration_s", fontsize=12)
    ax.set_title(
        "Step 10 — Duration Confound Check\n"
        "Blue = survives residualization  |  Red = loses significance\n"
        "Large dots = asymmetry features",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_survival_bar(
    stats_orig: pd.DataFrame,
    stats_resid: pd.DataFrame,
    path: Path,
) -> None:
    """Bar chart: how many features survive residualization at each threshold."""
    thresholds = [0.05, 0.01, 0.001]
    labels = ["p < 0.05", "p < 0.01", "p < 0.001"]

    orig_counts = [int((stats_orig["kw_p"] < t).sum()) for t in thresholds]
    resid_counts = [int((stats_resid["kw_p"] < t).sum()) for t in thresholds]
    pct_survived = [r / o * 100 if o > 0 else 0 for o, r in zip(orig_counts, resid_counts)]

    x = np.arange(len(thresholds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, orig_counts, width, label="Original", color="#55A868", alpha=0.8)
    bars2 = ax.bar(x + width / 2, resid_counts, width, label="After removing duration_s",
                   color="#4C72B0", alpha=0.8)

    # Annotate survival %
    for i, (o, r, pct) in enumerate(zip(orig_counts, resid_counts, pct_survived)):
        ax.text(i + width / 2, r + 1, f"{pct:.0f}%\nsurvive", ha="center",
                va="bottom", fontsize=9, color="#4C72B0", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of significant features")
    ax.set_title("Step 10 — Features Surviving Duration Residualization\n"
                 "High survival rate → features capture genuine gait patterns, not just trial length",
                 fontsize=11)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_before_after_boxplots(
    df_orig: pd.DataFrame,
    df_resid: pd.DataFrame,
    features: List[str],
    path: Path,
) -> None:
    """Side-by-side: original vs residualized boxplots for top features."""
    n = min(len(features), 6)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row_i, feat in enumerate(features[:n]):
        for col_i, (df, label) in enumerate([(df_orig, "Original"), (df_resid, "Residualized\n(duration removed)")]):
            ax = axes[row_i, col_i]
            data_by_group = [
                df.loc[df["group"] == g, feat].dropna().values
                for g in GROUPS
            ]
            bp = ax.boxplot(
                data_by_group,
                tick_labels=[f"{g}\n(n={len(d)})" for g, d in zip(GROUPS, data_by_group)],
                patch_artist=True, widths=0.5, showfliers=False,
            )
            for patch, g in zip(bp["boxes"], GROUPS):
                patch.set_facecolor(GROUP_COLORS[g])
                patch.set_alpha(0.55)
            for median in bp["medians"]:
                median.set_color("black")
                median.set_linewidth(2)

            # Overlay dots
            for i, (g, vals) in enumerate(zip(GROUPS, data_by_group)):
                jitter = np.random.default_rng(99).uniform(-0.13, 0.13, size=len(vals))
                ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                           color=GROUP_COLORS[g], alpha=0.45, s=14, edgecolors="none", zorder=3)

            # KW p-value on this data
            valid = [v for v in data_by_group if len(v) >= 3]
            if len(valid) >= 2:
                _, p = stats.kruskal(*valid)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                ax.set_title(f"{feat[:30]}\n{label}  [KW p={p:.2e} {sig}]", fontsize=9)
            else:
                ax.set_title(f"{feat[:30]}\n{label}", fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("Step 10 — Before/After Duration Residualization\n"
                 "Group separability before (left) vs after (right) removing trial-duration effect",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────


def run(
    base_path: str = "dataset/data",
    out_dir: str = "results",
) -> None:
    base = Path(base_path)
    out = Path(out_dir)
    ensure_dirs(out)
    art = out / "artifacts"
    fig_dir = out / "figures"

    print("=" * 60)
    print("STEP 10 — DURATION CONFOUND CHECK (RESIDUALIZATION)")
    print("=" * 60)
    print(f"Covariate: {COVARIATE}")

    # ── 1. Load data (same as directionality) ──
    trials = find_trials(base, limit=None)
    X_sensor, y, subj, tr_ids, pkey = collect_features(base, trials)

    asym_csv = art / "asymmetry_per_trial.csv"
    X_asym_cols: List[str] = []
    if asym_csv.exists():
        df_asym = pd.read_csv(asym_csv)
        asym_available = [c for c in ASYM_COLS if c in df_asym.columns]
        asym_trial_set = set(df_asym["trial"].values)
        asym_rows = []
        for tr_name in tr_ids:
            if tr_name in asym_trial_set:
                row = df_asym.loc[df_asym["trial"] == tr_name, asym_available]
                asym_rows.append(row.iloc[0].to_dict() if len(row) == 1 else {c: np.nan for c in asym_available})
            else:
                asym_rows.append({c: np.nan for c in asym_available})
        X_asym = pd.DataFrame(asym_rows, index=X_sensor.index)
        X_all = pd.concat([X_sensor, X_asym], axis=1)
        X_asym_cols = asym_available

    else:
        X_all = X_sensor.copy()

    # ── 2. Per-subject aggregation ──
    df_subj = build_subject_df(X_all, y, subj)
    feature_cols = [c for c in X_all.columns if c not in ("group", "subject")]
    print(f"Subjects: {len(df_subj)}  |  Features: {len(feature_cols)}")

    # Show duration distribution
    dur = df_subj[COVARIATE]
    for g in GROUPS:
        m = df_subj.loc[df_subj["group"] == g, COVARIATE].median()
        print(f"  {g} median duration: {m:.1f} s")

    # ── 3. Original stats ──
    print("\nComputing original Kruskal-Wallis stats...")
    orig_stats = pd.DataFrame([feature_stats(df_subj, f) for f in feature_cols])

    # ── 4. Residualize ──
    print(f"Residualizing {len(feature_cols)-1} features against {COVARIATE}...")
    df_resid = residualize(df_subj, feature_cols, covariate=COVARIATE)

    # ── 5. Residualized stats ──
    print("Computing Kruskal-Wallis stats on residuals...")
    resid_stats = pd.DataFrame([feature_stats(df_resid, f) for f in feature_cols])

    # ── 6. Survival summary ──
    n_orig_05 = int((orig_stats["kw_p"] < 0.05).sum())
    n_resid_05 = int((resid_stats["kw_p"] < 0.05).sum())
    n_orig_001 = int((orig_stats["kw_p"] < 0.001).sum())
    n_resid_001 = int((resid_stats["kw_p"] < 0.001).sum())

    pct_05 = 100 * n_resid_05 / max(n_orig_05, 1)
    pct_001 = 100 * n_resid_001 / max(n_orig_001, 1)

    print(f"\n{'─' * 50}")
    print(f"  p<0.05:  {n_orig_05} → {n_resid_05}  ({pct_05:.0f}% survive)")
    print(f"  p<0.001: {n_orig_001} → {n_resid_001}  ({pct_001:.0f}% survive)")

    # Which features survive?
    survived = resid_stats[resid_stats["kw_p"] < 0.05]["feature"].tolist()
    dropped = orig_stats[
        (orig_stats["kw_p"] < 0.05) &
        ~orig_stats["feature"].isin(survived)
    ]["feature"].tolist()

    # Asymmetry features check
    asym_orig_sig = [f for f in X_asym_cols if f in orig_stats[orig_stats["kw_p"] < 0.05]["feature"].values]
    asym_survived = [f for f in asym_orig_sig if f in survived]
    print(f"\n  Asymmetry features: {len(asym_orig_sig)} originally significant, {len(asym_survived)} survive")

    # ── 7. Detailed comparison table ──
    merged = orig_stats[["feature", "kw_p", "direction"]].merge(
        resid_stats[["feature", "kw_p"]].rename(columns={"kw_p": "kw_p_resid"}),
        on="feature",
    ).sort_values("kw_p")

    print(f"\n{'─' * 85}")
    print(f"{'Feature':38s} {'Direction':14s} {'Orig p':>12s} {'Resid p':>12s} {'Verdict':>12s}")
    print(f"{'─' * 85}")
    for _, r in merged.head(30).iterrows():
        verdict = "SURVIVES ✓" if r["kw_p_resid"] < 0.05 else "drops ✗"
        feat = str(r["feature"])[:36]
        print(
            f"  {feat:36s} {r['direction']:14s} "
            f"{r['kw_p']:12.2e} {r['kw_p_resid']:12.2e}  {verdict}"
        )

    # ── 8. Plots ──
    print("\nGenerating figures...")

    plot_pvalue_scatter(
        orig_stats, resid_stats,
        fig_dir / "step10_duration_scatter.png",
    )

    plot_survival_bar(
        orig_stats, resid_stats,
        fig_dir / "step10_duration_survival_bar.png",
    )

    # Before/after boxplots for top surviving features (excluding duration itself)
    top_survivors = (
        resid_stats[resid_stats["feature"] != COVARIATE]
        .sort_values("kw_p")
        .head(6)["feature"]
        .tolist()
    )
    plot_before_after_boxplots(
        df_subj, df_resid, top_survivors,
        fig_dir / "step10_before_after_boxplots.png",
    )

    # Asymmetry features before/after
    if asym_orig_sig:
        plot_before_after_boxplots(
            df_subj, df_resid,
            asym_orig_sig[:6],
            fig_dir / "step10_asymmetry_before_after.png",
        )

    # ── 9. Save artifacts ──
    merged.to_csv(art / "duration_check.csv", index=False)

    result_json = {
        "covariate": COVARIATE,
        "n_subjects": len(df_subj),
        "duration_medians": {
            g: round(float(df_subj.loc[df_subj["group"] == g, COVARIATE].median()), 2)
            for g in GROUPS
        },
        "original_significant": {
            "p05": n_orig_05,
            "p001": n_orig_001,
        },
        "after_residualization": {
            "p05": n_resid_05,
            "p001": n_resid_001,
        },
        "survival_rate_p05_pct": round(pct_05, 1),
        "survival_rate_p001_pct": round(pct_001, 1),
        "asymmetry_features_originally_significant": asym_orig_sig,
        "asymmetry_features_surviving": asym_survived,
        "features_losing_significance": dropped[:20],
        "top_survivors": [
            {
                "feature": r["feature"],
                "direction": r["direction"],
                "original_p": float(r["kw_p"]),
                "residualized_p": float(r["kw_p_resid"]),
            }
            for _, r in merged[merged["kw_p_resid"] < 0.001]
            .head(20)
            .iterrows()
        ],
    }
    (art / "duration_check.json").write_text(
        json.dumps(result_json, indent=2), encoding="utf-8"
    )

    print(f"\nDone.")
    print(f"  Figures: step10_*.png")
    print(f"  Artifacts: {art / 'duration_check.json'}")


if __name__ == "__main__":
    run()
