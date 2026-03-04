"""Feature directionality and cross-subject consistency analysis.

Answers the professor's core questions:
  1. For each discriminative feature, does it increase or decrease
     from healthy → ortho → neuro?
  2. Is the direction consistent across subjects (not just on average)?
  3. Visualize with per-feature boxplots.

Usage:
    python analysis/directionality.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

# Group ordering (the professor wants to see this gradient)
GROUPS = ["healthy", "ortho", "neuro"]
GROUP_COLORS = {"healthy": "#55A868", "ortho": "#4C72B0", "neuro": "#C44E52"}

# Asymmetry feature columns
ASYM_COLS = [
    "stride_absAI", "step_absAI", "stride_abs_diff", "step_abs_diff",
    "stride_AI", "step_AI", "step_CV_l", "step_CV_r",
    "stride_CV_l", "stride_CV_r", "mean_step_time",
]


# ── Per-subject aggregation ──────────────────────────────────────────


def build_subject_df(
    X: pd.DataFrame,
    y: pd.Series,
    subj: pd.Series,
) -> pd.DataFrame:
    """Aggregate trial-level features to subject-level (mean per subject)."""
    df = X.copy()
    df["group"] = y.values
    df["subject"] = subj.values
    # Mean across trials per subject
    grouped = df.groupby("subject")
    X_subj = grouped[X.columns].mean()
    # Attach group label (take first — all trials same subject same group)
    X_subj["group"] = grouped["group"].first()
    return X_subj.reset_index()


# ── Directionality classification ────────────────────────────────────


def classify_direction(
    medians: Dict[str, float],
) -> str:
    """Classify the H→O→N trend as ↑, ↓, or mixed."""
    h, o, n = medians["healthy"], medians["ortho"], medians["neuro"]
    if h <= o <= n and not (h == o == n):
        return "↑ H<O<N"
    if h >= o >= n and not (h == o == n):
        return "↓ H>O>N"
    if h < o and o > n:
        return "∧ peak-ortho"
    if h > o and o < n:
        return "∨ valley-ortho"
    if h == o == n:
        return "— flat"
    return "~ mixed"


# ── Statistical tests ────────────────────────────────────────────────


def feature_stats(
    df_subj: pd.DataFrame,
    feature: str,
) -> Dict:
    """Compute group stats, Kruskal-Wallis, and pairwise Mann-Whitney."""
    result: Dict = {"feature": feature}

    groups_data = {}
    for g in GROUPS:
        vals = df_subj.loc[df_subj["group"] == g, feature].dropna().values
        groups_data[g] = vals
        result[f"{g}_median"] = float(np.median(vals)) if len(vals) > 0 else np.nan
        result[f"{g}_mean"] = float(np.mean(vals)) if len(vals) > 0 else np.nan
        result[f"{g}_std"] = float(np.std(vals)) if len(vals) > 0 else np.nan
        result[f"{g}_n"] = len(vals)

    # Direction
    medians = {g: result[f"{g}_median"] for g in GROUPS}
    result["direction"] = classify_direction(medians)

    # Kruskal-Wallis (non-parametric 3-group test)
    valid = [v for v in groups_data.values() if len(v) >= 3]
    if len(valid) >= 2:
        stat, p = stats.kruskal(*valid)
        result["kw_stat"] = float(stat)
        result["kw_p"] = float(p)
    else:
        result["kw_stat"] = np.nan
        result["kw_p"] = np.nan

    # Pairwise Mann-Whitney U + rank-biserial r (effect size)
    pairs = [("healthy", "ortho"), ("healthy", "neuro"), ("ortho", "neuro")]
    for g1, g2 in pairs:
        a, b = groups_data[g1], groups_data[g2]
        if len(a) >= 3 and len(b) >= 3:
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            # Rank-biserial r = 1 - 2U/(n1*n2)
            r = 1.0 - 2.0 * u / (len(a) * len(b))
            result[f"{g1}_vs_{g2}_p"] = float(p)
            result[f"{g1}_vs_{g2}_r"] = float(r)
        else:
            result[f"{g1}_vs_{g2}_p"] = np.nan
            result[f"{g1}_vs_{g2}_r"] = np.nan

    return result


# ── Cross-subject consistency ────────────────────────────────────────


def consistency_score(
    df_subj: pd.DataFrame,
    feature: str,
) -> Dict:
    """Measure what fraction of subjects follow the group-level direction.

    For each pair (e.g., healthy vs neuro), if the group-level direction
    says healthy > neuro, compute what % of neuro subjects actually have
    values below the healthy median. Higher = more consistent.
    """
    result: Dict = {"feature": feature}

    groups_data = {}
    for g in GROUPS:
        groups_data[g] = df_subj.loc[df_subj["group"] == g, feature].dropna().values

    healthy_med = np.median(groups_data["healthy"]) if len(groups_data["healthy"]) > 0 else np.nan
    neuro_vals = groups_data["neuro"]
    ortho_vals = groups_data["ortho"]

    if np.isnan(healthy_med):
        return result

    # Healthy vs Neuro: what % of neuro follow the expected direction?
    h_med = float(healthy_med)
    h_mean = float(np.mean(groups_data["healthy"])) if len(groups_data["healthy"]) > 0 else h_med
    n_mean = float(np.mean(neuro_vals)) if len(neuro_vals) > 0 else h_med

    if len(neuro_vals) > 0:
        if h_mean > n_mean:
            # Expect neuro < healthy median
            frac = float(np.mean(neuro_vals < h_med))
            result["h_vs_n_direction"] = "H > N"
            result["h_vs_n_concordance"] = frac
        else:
            # Expect neuro > healthy median
            frac = float(np.mean(neuro_vals > h_med))
            result["h_vs_n_direction"] = "H < N"
            result["h_vs_n_concordance"] = frac

    # Healthy vs Ortho
    o_mean = float(np.mean(ortho_vals)) if len(ortho_vals) > 0 else h_med
    if len(ortho_vals) > 0:
        if h_mean > o_mean:
            frac = float(np.mean(ortho_vals < h_med))
            result["h_vs_o_direction"] = "H > O"
            result["h_vs_o_concordance"] = frac
        else:
            frac = float(np.mean(ortho_vals > h_med))
            result["h_vs_o_direction"] = "H < O"
            result["h_vs_o_concordance"] = frac

    return result


# ── Boxplot grid ─────────────────────────────────────────────────────


def plot_boxplot_grid(
    df_subj: pd.DataFrame,
    features: List[str],
    title: str,
    path: Path,
    ncols: int = 4,
) -> None:
    """Draw a grid of boxplots, one per feature, with H/O/N groups."""
    n = len(features)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, feat in enumerate(features):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        data_by_group = []
        positions = []
        for i, g in enumerate(GROUPS):
            vals = df_subj.loc[df_subj["group"] == g, feat].dropna().values
            data_by_group.append(vals)
            positions.append(i)

        bp = ax.boxplot(
            data_by_group,
            positions=positions,
            tick_labels=GROUPS,
            patch_artist=True,
            widths=0.6,
            showfliers=True,
            flierprops=dict(marker=".", markersize=3, alpha=0.4),
        )

        for patch, g in zip(bp["boxes"], GROUPS):
            patch.set_facecolor(GROUP_COLORS[g])
            patch.set_alpha(0.6)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(2)

        # Mark medians with text
        for i, g in enumerate(GROUPS):
            med_val = np.median(data_by_group[i]) if len(data_by_group[i]) > 0 else 0
            ax.text(
                i, med_val, f"{med_val:.3f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold",
            )

        # Shorten feature name for display
        short = feat.replace("__", "\n")
        if len(short) > 35:
            short = short[:32] + "..."
        ax.set_title(short, fontsize=9)
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Individual feature boxplot (larger, with scatter) ────────────────


def plot_single_boxplot(
    df_subj: pd.DataFrame,
    feature: str,
    direction: str,
    kw_p: float,
    consistency_info: Dict,
    path: Path,
) -> None:
    """Detailed single-feature boxplot with individual subject dots."""
    fig, ax = plt.subplots(figsize=(7, 5))

    data_by_group = []
    for g in GROUPS:
        vals = df_subj.loc[df_subj["group"] == g, feature].dropna().values
        data_by_group.append(vals)

    bp = ax.boxplot(
        data_by_group,
        positions=[0, 1, 2],
        tick_labels=[f"{g}\n(n={len(d)})" for g, d in zip(GROUPS, data_by_group)],
        patch_artist=True,
        widths=0.5,
        showfliers=False,
    )
    for patch, g in zip(bp["boxes"], GROUPS):
        patch.set_facecolor(GROUP_COLORS[g])
        patch.set_alpha(0.5)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2.5)

    # Overlay individual subject points (jittered)
    for i, (g, vals) in enumerate(zip(GROUPS, data_by_group)):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            color=GROUP_COLORS[g],
            alpha=0.5,
            s=18,
            edgecolors="none",
            zorder=3,
        )

    # Draw trend line through medians
    meds = [np.median(d) if len(d) > 0 else np.nan for d in data_by_group]
    ax.plot([0, 1, 2], meds, "k--", linewidth=1.5, alpha=0.7, zorder=4)

    # Annotations
    p_str = f"p={kw_p:.1e}" if kw_p < 0.001 else f"p={kw_p:.4f}"
    sig = "***" if kw_p < 0.001 else "**" if kw_p < 0.01 else "*" if kw_p < 0.05 else "ns"

    # Consistency annotation
    conc_str = ""
    h_vs_n = consistency_info.get("h_vs_n_concordance")
    if h_vs_n is not None:
        conc_str = f"  |  H-vs-N concordance: {h_vs_n:.0%}"

    ax.set_title(
        f"{feature}\n"
        f"Direction: {direction}  |  Kruskal-Wallis {p_str} {sig}{conc_str}",
        fontsize=10,
    )
    ax.set_ylabel("Feature value (subject-level mean)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── Summary table plot ───────────────────────────────────────────────


def plot_summary_table(
    stats_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    path: Path,
    top_n: int = 25,
) -> None:
    """Create a visual summary table of directionality and consistency."""
    merged = stats_df.merge(consistency_df, on="feature", how="left")
    merged = merged.sort_values("kw_p").head(top_n)

    fig, ax = plt.subplots(figsize=(18, 0.5 * top_n + 2))
    ax.axis("off")

    headers = [
        "Feature", "Direction",
        "Healthy\nmedian", "Ortho\nmedian", "Neuro\nmedian",
        "KW p-value", "Sig",
        "H-vs-N\nconcord.",
    ]
    rows = []
    colors = []
    for _, r in merged.iterrows():
        sig = "***" if r["kw_p"] < 0.001 else "**" if r["kw_p"] < 0.01 else "*" if r["kw_p"] < 0.05 else "ns"
        conc = f"{r.get('h_vs_n_concordance', 0):.0%}" if pd.notna(r.get("h_vs_n_concordance")) else "—"
        feat_name = str(r["feature"])
        if len(feat_name) > 35:
            feat_name = feat_name[:32] + "..."
        rows.append([
            feat_name,
            r["direction"],
            f"{r['healthy_median']:.4f}",
            f"{r['ortho_median']:.4f}",
            f"{r['neuro_median']:.4f}",
            f"{r['kw_p']:.2e}" if r["kw_p"] < 0.001 else f"{r['kw_p']:.4f}",
            sig,
            conc,
        ])
        # Row color based on direction consistency
        d = r["direction"]
        if "↑" in d or "↓" in d:
            colors.append(["#E8F5E9"] * len(headers))  # green = monotonic
        elif "∧" in d or "∨" in d:
            colors.append(["#FFF3E0"] * len(headers))  # orange = non-monotonic
        else:
            colors.append(["#FFFFFF"] * len(headers))

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellColours=colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Header styling
    for j in range(len(headers)):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        f"Feature Directionality Summary (Top {top_n} by Kruskal-Wallis p-value)\n"
        "Green rows = monotonic trend (↑ or ↓) across Healthy → Ortho → Neuro",
        fontsize=12, fontweight="bold", pad=20,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────


def run(
    base_path: str = "dataset/data",
    out_dir: str = "results",
    top_n_boxplots: int = 20,
    top_n_individual: int = 12,
) -> None:
    base = Path(base_path)
    out = Path(out_dir)
    ensure_dirs(out)
    art = out / "artifacts"
    fig_dir = out / "figures"

    print("=" * 60)
    print("STEP 09 — FEATURE DIRECTIONALITY & CONSISTENCY ANALYSIS")
    print("=" * 60)

    # ── 1. Load sensor features ──
    trials = find_trials(base, limit=None)
    X_sensor, y, subj, tr_ids, pkey = collect_features(base, trials)
    print(f"Loaded {X_sensor.shape[1]} sensor features, {len(y)} trials")

    # ── 2. Merge asymmetry features ──
    asym_csv = art / "asymmetry_per_trial.csv"
    X_asym_cols = []
    if asym_csv.exists():
        df_asym = pd.read_csv(asym_csv)
        asym_available = [c for c in ASYM_COLS if c in df_asym.columns]
        asym_trial_set = set(df_asym["trial"].values)
        asym_rows = []
        for tr_name in tr_ids:
            if tr_name in asym_trial_set:
                row = df_asym.loc[df_asym["trial"] == tr_name, asym_available]
                if len(row) == 1:
                    asym_rows.append(row.iloc[0].to_dict())
                else:
                    asym_rows.append({c: np.nan for c in asym_available})
            else:
                asym_rows.append({c: np.nan for c in asym_available})
        X_asym = pd.DataFrame(asym_rows, index=X_sensor.index)
        X_all = pd.concat([X_sensor, X_asym], axis=1)
        X_asym_cols = asym_available
        print(f"Merged {len(asym_available)} asymmetry features")
    else:
        X_all = X_sensor.copy()
        print("No asymmetry CSV found, using sensor features only")

    # ── 3. Per-subject aggregation ──
    df_subj = build_subject_df(X_all, y, subj)
    n_subj = len(df_subj)
    print(f"Aggregated to {n_subj} subjects")
    for g in GROUPS:
        n = (df_subj["group"] == g).sum()
        print(f"  {g}: {n} subjects")

    # ── 4. Compute stats for every feature ──
    feature_cols = [c for c in X_all.columns if c not in ("group", "subject")]
    print(f"\nAnalyzing {len(feature_cols)} features...")

    all_stats = []
    all_consistency = []
    for feat in feature_cols:
        s = feature_stats(df_subj, feat)
        c = consistency_score(df_subj, feat)
        all_stats.append(s)
        all_consistency.append(c)

    stats_df = pd.DataFrame(all_stats)
    consist_df = pd.DataFrame(all_consistency)

    # ── 5. Sort by significance ──
    stats_df = stats_df.sort_values("kw_p")
    n_sig_001 = (stats_df["kw_p"] < 0.001).sum()
    n_sig_01 = (stats_df["kw_p"] < 0.01).sum()
    n_sig_05 = (stats_df["kw_p"] < 0.05).sum()
    print(f"\nSignificant features: {n_sig_001} (p<0.001), {n_sig_01} (p<0.01), {n_sig_05} (p<0.05)")

    # Direction summary
    dir_counts = stats_df.loc[stats_df["kw_p"] < 0.05, "direction"].value_counts()
    print(f"\nDirection distribution (p<0.05 features):")
    for d, count in dir_counts.items():
        print(f"  {d}: {count}")

    # ── 6. Print directionality table (top features) ──
    top = stats_df.head(30)
    merged_top = top.merge(consist_df, on="feature", how="left")

    print(f"\n{'─' * 100}")
    print(f"{'Feature':40s} {'Direction':15s} {'H median':>10s} {'O median':>10s} {'N median':>10s} {'KW p':>12s} {'H-N conc':>10s}")
    print(f"{'─' * 100}")
    for _, r in merged_top.iterrows():
        sig = "***" if r["kw_p"] < 0.001 else "** " if r["kw_p"] < 0.01 else "*  " if r["kw_p"] < 0.05 else "ns "
        feat = str(r["feature"])[:38]
        conc = f"{r.get('h_vs_n_concordance', 0):.0%}" if pd.notna(r.get("h_vs_n_concordance")) else "—"
        print(
            f"  {feat:38s} {r['direction']:15s} "
            f"{r['healthy_median']:10.4f} {r['ortho_median']:10.4f} {r['neuro_median']:10.4f} "
            f"{r['kw_p']:12.2e} {sig} {conc:>8s}"
        )

    # ── 7. Boxplot grid (top features by KW p-value) ──
    top_features = stats_df.head(top_n_boxplots)["feature"].tolist()
    print(f"\nPlotting boxplot grid for top {len(top_features)} features...")

    plot_boxplot_grid(
        df_subj, top_features,
        "Step 09 — Feature Directionality: Top 20 by Kruskal-Wallis p-value\n"
        "(Subject-level, Green=Healthy, Blue=Ortho, Red=Neuro)",
        fig_dir / "step09_directionality_grid.png",
    )

    # Separate grid for asymmetry features
    asym_in_data = [c for c in X_asym_cols if c in stats_df["feature"].values]
    if asym_in_data:
        plot_boxplot_grid(
            df_subj, asym_in_data,
            "Step 09 — Asymmetry Feature Directionality\n"
            "(Subject-level, Green=Healthy, Blue=Ortho, Red=Neuro)",
            fig_dir / "step09_directionality_asymmetry.png",
            ncols=4,
        )

    # ── 8. Individual detailed boxplots (top features) ──
    top_individual = stats_df.head(top_n_individual)["feature"].tolist()
    print(f"Plotting {len(top_individual)} individual boxplots...")

    for feat in top_individual:
        row = stats_df.loc[stats_df["feature"] == feat].iloc[0]
        c_row = consist_df.loc[consist_df["feature"] == feat]
        c_info = c_row.iloc[0].to_dict() if len(c_row) > 0 else {}

        safe_name = feat.replace("/", "_").replace("\\", "_")[:50]
        plot_single_boxplot(
            df_subj, feat,
            row["direction"], row["kw_p"], c_info,
            fig_dir / f"step09_box_{safe_name}.png",
        )

    # ── 9. Summary table figure ──
    plot_summary_table(stats_df, consist_df, fig_dir / "step09_summary_table.png", top_n=25)

    # ── 10. Save artifacts ──
    # Full stats table
    stats_df.to_csv(art / "directionality_stats.csv", index=False)
    # Consistency table
    merged_all = stats_df.merge(consist_df, on="feature", how="left")
    merged_all.to_csv(art / "directionality_full.csv", index=False)

    # JSON summary of top features
    summary = {
        "total_features": len(feature_cols),
        "significant_p001": int(n_sig_001),
        "significant_p01": int(n_sig_01),
        "significant_p05": int(n_sig_05),
        "direction_counts_sig05": {str(k): int(v) for k, v in dir_counts.items()},
        "top_features": [],
    }
    for _, r in merged_top.head(20).iterrows():
        entry = {
            "feature": r["feature"],
            "direction": r["direction"],
            "healthy_median": round(float(r["healthy_median"]), 6),
            "ortho_median": round(float(r["ortho_median"]), 6),
            "neuro_median": round(float(r["neuro_median"]), 6),
            "kw_p": float(r["kw_p"]),
            "h_vs_n_concordance": round(float(r.get("h_vs_n_concordance", 0)), 3)
            if pd.notna(r.get("h_vs_n_concordance"))
            else None,
        }
        summary["top_features"].append(entry)
    (art / "directionality_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # ── 11. Consistency highlight ──
    print(f"\n{'=' * 60}")
    print("CONSISTENCY HIGHLIGHTS (top significant features)")
    print("=" * 60)
    sig_merged = merged_all[merged_all["kw_p"] < 0.05].sort_values("kw_p")
    for _, r in sig_merged.head(20).iterrows():
        feat = str(r["feature"])[:38]
        d = r["direction"]
        conc = r.get("h_vs_n_concordance")
        conc_str = f"{conc:.0%}" if pd.notna(conc) else "—"
        # Classify consistency
        if pd.notna(conc):
            if conc >= 0.7:
                tag = "STRONG"
            elif conc >= 0.5:
                tag = "moderate"
            else:
                tag = "weak"
        else:
            tag = "—"
        print(f"  {feat:38s} {d:15s} concordance={conc_str:>6s}  [{tag}]")

    print(f"\nDone. Artifacts: {art / 'directionality_*.csv|json'}")
    print(f"Figures: {fig_dir / 'step09_*.png'}")


if __name__ == "__main__":
    run()
