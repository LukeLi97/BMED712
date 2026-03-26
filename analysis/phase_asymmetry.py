"""Phase-specific gait asymmetry analysis.

Computes stride/step asymmetry separately for three walking phases:
  - pre_uturn  : steady-state approach (before U-turn)
  - uturn      : U-turn maneuver
  - post_uturn : steady-state return (after U-turn)

Answers: Does the U-turn challenge reveal pathology better than flat walking?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.asymmetry import (  # type: ignore
    extract_heel_strikes,
    _step_times,
    _stride_times,
    _cv,
)
from analysis.pipeline import ensure_dirs, find_trials  # type: ignore
from dataset.quick_start.load_data import load_trial  # type: ignore

MIN_HS = 3          # minimum heel strikes per side per phase
GROUP_ORDER = ["healthy", "ortho", "neuro"]
GROUP_COLORS = {"healthy": "#4CAF50", "ortho": "#FF9800", "neuro": "#F44336"}
PHASES = ["pre_uturn", "uturn", "post_uturn"]


# ── heel-strike phase splitting ──────────────────────────────────────


def _split_hs_by_phase(
    hs: np.ndarray, uturn: list
) -> Dict[str, np.ndarray]:
    """Split sorted heel-strike indices into pre/uturn/post segments."""
    if len(uturn) < 2:
        return {"pre_uturn": hs, "uturn": np.array([], dtype=int), "post_uturn": np.array([], dtype=int)}
    u0, u1 = int(uturn[0]), int(uturn[1])
    return {
        "pre_uturn":  hs[hs < u0],
        "uturn":      hs[(hs >= u0) & (hs <= u1)],
        "post_uturn": hs[hs > u1],
    }


def _phase_metrics(
    left_hs: np.ndarray, right_hs: np.ndarray, freq: float
) -> Optional[Dict]:
    """Compute asymmetry metrics for one phase. Returns None if insufficient data."""
    if left_hs.size < MIN_HS or right_hs.size < MIN_HS:
        return None

    step_l, step_r = _step_times(left_hs, right_hs, freq)
    stride_l = _stride_times(left_hs, freq)
    stride_r = _stride_times(right_hs, freq)

    if step_l.size == 0 or step_r.size == 0:
        return None

    mean_sl, mean_sr = float(np.mean(step_l)), float(np.mean(step_r))
    mean_tl = float(np.mean(stride_l)) if stride_l.size else float("nan")
    mean_tr = float(np.mean(stride_r)) if stride_r.size else float("nan")

    def ai(a, b):
        d = (a + b) / 2
        return (a - b) / d if d > 0 else float("nan")

    return {
        "mean_step_l": mean_sl,
        "mean_step_r": mean_sr,
        "step_AI": ai(mean_sl, mean_sr),
        "step_absAI": abs(ai(mean_sl, mean_sr)),
        "step_abs_diff": abs(mean_sl - mean_sr),
        "step_CV_l": _cv(step_l),
        "step_CV_r": _cv(step_r),
        "mean_stride_l": mean_tl,
        "mean_stride_r": mean_tr,
        "stride_AI": ai(mean_tl, mean_tr),
        "stride_absAI": abs(ai(mean_tl, mean_tr)),
        "stride_abs_diff": abs(mean_tl - mean_tr),
        "n_steps_l": int(step_l.size),
        "n_steps_r": int(step_r.size),
        "n_strides_l": int(stride_l.size),
        "n_strides_r": int(stride_r.size),
    }


# ── main extraction ──────────────────────────────────────────────────


def extract_all_phases(base_path: str) -> pd.DataFrame:
    """Return long-format DataFrame: one row per (trial, phase)."""
    rows: List[Dict] = []
    for trial_name in find_trials(Path(base_path)):
        try:
            trial = load_trial(base_path, trial_name)
        except Exception:
            continue
        md = trial.get("metadata", {})
        freq = float(md.get("freq", 100.0))
        uturn = md.get("uturnBoundaries", [])
        group = str(md.get("group", "unknown"))
        subject = str(md.get("subject", ""))
        path_key = str(md.get("pathologyKey", ""))

        left_all = extract_heel_strikes(md.get("leftGaitEvents", []))
        right_all = extract_heel_strikes(md.get("rightGaitEvents", []))

        left_phases = _split_hs_by_phase(left_all, uturn)
        right_phases = _split_hs_by_phase(right_all, uturn)

        for phase in PHASES:
            m = _phase_metrics(left_phases[phase], right_phases[phase], freq)
            if m is None:
                continue
            rows.append({
                "trial": trial_name,
                "subject": subject,
                "group": group,
                "pathology_key": path_key,
                "phase": phase,
                **m,
            })

    return pd.DataFrame(rows)


# ── statistics ───────────────────────────────────────────────────────


def phase_stats(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """KW test + pairwise MW per phase. Returns summary DataFrame."""
    results = []
    for phase in PHASES:
        sub = df[df["phase"] == phase].dropna(subset=[metric])
        groups = {g: sub.loc[sub["group"] == g, metric].values for g in GROUP_ORDER}
        arrs = [v for v in groups.values() if len(v) >= 3]
        if len(arrs) < 2:
            continue
        kw = stats.kruskal(*arrs)
        # Pairwise H vs N (most clinically relevant)
        h = groups.get("healthy", np.array([]))
        n = groups.get("neuro", np.array([]))
        o = groups.get("ortho", np.array([]))
        mw_hn = stats.mannwhitneyu(h, n, alternative="two-sided") if h.size >= 3 and n.size >= 3 else None
        mw_ho = stats.mannwhitneyu(h, o, alternative="two-sided") if h.size >= 3 and o.size >= 3 else None

        def cohens_d(a, b):
            if len(a) < 2 or len(b) < 2:
                return float("nan")
            pooled = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
            return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else float("nan")

        results.append({
            "phase": phase,
            "metric": metric,
            "n_healthy": int(h.size),
            "n_ortho": int(o.size),
            "n_neuro": int(n.size),
            "median_healthy": float(np.median(h)) if h.size else float("nan"),
            "median_ortho": float(np.median(o)) if o.size else float("nan"),
            "median_neuro": float(np.median(n)) if n.size else float("nan"),
            "kw_p": float(kw.pvalue),
            "mw_p_H_vs_N": float(mw_hn.pvalue) if mw_hn else float("nan"),
            "mw_p_H_vs_O": float(mw_ho.pvalue) if mw_ho else float("nan"),
            "cohens_d_H_vs_N": cohens_d(h, n),
        })
    return pd.DataFrame(results)


# ── plotting ─────────────────────────────────────────────────────────


def plot_phase_comparison(
    df: pd.DataFrame,
    metric: str,
    title: str,
    path: Path,
    stats_df: pd.DataFrame,
):
    """3-panel boxplot (one panel per phase) for a given metric."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    phase_labels = {"pre_uturn": "Pre U-turn\n(steady-state)", "uturn": "U-turn\n(maneuver)", "post_uturn": "Post U-turn\n(recovery)"}

    for ax, phase in zip(axes, PHASES):
        sub = df[df["phase"] == phase]
        data = [sub.loc[sub["group"] == g, metric].dropna().values for g in GROUP_ORDER]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=2))
        for patch, g in zip(bp["boxes"], GROUP_ORDER):
            patch.set_facecolor(GROUP_COLORS[g])
            patch.set_alpha(0.75)

        ax.set_title(phase_labels.get(phase, phase), fontsize=11)
        ax.set_xticklabels(["Healthy", "Ortho", "Neuro"], fontsize=9)
        ax.set_xlabel("Group")
        if ax == axes[0]:
            ax.set_ylabel(metric.replace("_", " "), fontsize=10)

        # Annotate KW p-value
        row = stats_df[(stats_df["phase"] == phase) & (stats_df["metric"] == metric)]
        if not row.empty:
            p = float(row["kw_p"].iloc[0])
            d = float(row["cohens_d_H_vs_N"].iloc[0])
            stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            ax.set_xlabel(f"KW {stars}  |d|={abs(d):.2f}", fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_phase_heatmap(stats_df: pd.DataFrame, path: Path):
    """Heatmap: metric × phase, colour = Cohen's d (H vs N)."""
    metrics = stats_df["metric"].unique()
    pivot_d = stats_df.pivot(index="metric", columns="phase", values="cohens_d_H_vs_N").reindex(columns=PHASES)
    pivot_p = stats_df.pivot(index="metric", columns="phase", values="kw_p").reindex(columns=PHASES)

    fig, ax = plt.subplots(figsize=(8, max(4, len(metrics) * 0.55)))
    im = ax.imshow(pivot_d.values.astype(float), aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1.2)
    plt.colorbar(im, ax=ax, label="Cohen's d (Healthy vs Neuro)")

    phase_labels = ["Pre U-turn", "U-turn", "Post U-turn"]
    ax.set_xticks(range(3))
    ax.set_xticklabels(phase_labels, fontsize=10)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m.replace("_", " ") for m in pivot_d.index], fontsize=9)
    ax.set_title("Effect Size (Cohen's d) per Phase — Healthy vs Neuro", fontsize=12, fontweight="bold")

    for i, metric in enumerate(pivot_d.index):
        for j, phase in enumerate(PHASES):
            d_val = pivot_d.loc[metric, phase]
            p_val = pivot_p.loc[metric, phase]
            if pd.isna(d_val):
                continue
            stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
            ax.text(j, i, f"{d_val:.2f}{stars}", ha="center", va="center", fontsize=8, color="black")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_delta_phase(df_subj: pd.DataFrame, metric: str, path: Path):
    """Bar chart: change in median asymmetry from pre_uturn → uturn → post_uturn per group."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    width = 0.25

    for i, g in enumerate(GROUP_ORDER):
        sub = df_subj[df_subj["group"] == g]
        medians = []
        for phase in PHASES:
            vals = sub.loc[sub["phase"] == phase, metric].dropna()
            medians.append(float(np.median(vals)) if len(vals) > 0 else float("nan"))
        ax.bar(x + (i - 1) * width, medians, width, label=g.capitalize(),
               color=GROUP_COLORS[g], alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(["Pre U-turn", "U-turn", "Post U-turn"])
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"Phase-wise Median {metric.replace('_', ' ')} by Group", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ── entry point ──────────────────────────────────────────────────────


def run(base_path: str, out_dir: str):
    out = Path(out_dir)
    ensure_dirs(out)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    art_dir = out / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    print("[phase_asymmetry] Extracting per-phase heel strikes …")
    df = extract_all_phases(base_path)

    if df.empty:
        print("[phase_asymmetry] No data extracted — check dataset path.")
        return

    df.to_csv(art_dir / "phase_asymmetry.csv", index=False)
    print(f"[phase_asymmetry] Saved {len(df)} rows → artifacts/phase_asymmetry.csv")

    # Subject-level means (avoid trial-level pseudoreplication)
    df_subj = (
        df.groupby(["subject", "group", "pathology_key", "phase"])
        .mean(numeric_only=True)
        .reset_index()
    )

    KEY_METRICS = ["stride_absAI", "step_absAI", "stride_abs_diff", "step_CV_l"]

    all_stats = []
    for metric in KEY_METRICS:
        s = phase_stats(df_subj, metric)
        all_stats.append(s)
        # 3-panel boxplot
        plot_phase_comparison(
            df_subj, metric,
            title=f"{metric.replace('_', ' ')} across Walking Phases",
            path=fig_dir / f"step12_phase_{metric}.png",
            stats_df=s,
        )
        # delta bar chart
        plot_delta_phase(df_subj, metric, fig_dir / f"step12_delta_{metric}.png")

    stats_df = pd.concat(all_stats, ignore_index=True)
    stats_df.to_csv(art_dir / "phase_asymmetry_stats.csv", index=False)

    # Heatmap
    plot_phase_heatmap(stats_df, fig_dir / "step12_phase_heatmap.png")

    # Print summary
    print("\n[phase_asymmetry] === Summary (stride_absAI) ===")
    print(stats_df[stats_df["metric"] == "stride_absAI"][
        ["phase", "median_healthy", "median_ortho", "median_neuro", "kw_p", "cohens_d_H_vs_N"]
    ].to_string(index=False))
    print(f"\n[phase_asymmetry] Figures saved → {fig_dir}/step12_*.png")


if __name__ == "__main__":
    base = str(REPO_ROOT / "dataset" / "data")
    out = str(REPO_ROOT / "results")
    run(base, out)
