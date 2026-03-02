"""Extended asymmetry analysis: subtype breakdown, ROC classifier,
clinical-score correlation.

Reads the CSVs produced by analysis/asymmetry.py and metadata from
the dataset to generate additional figures and statistics.

Usage:
    python analysis/asymmetry_extended.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.pipeline import ensure_dirs, find_trials
from dataset.quick_start.load_data import load_trial


# ── helpers ──────────────────────────────────────────────────────────

def _sig(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return 0.0
    n1, n2 = a.size, b.size
    sp2 = ((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / max(
        n1 + n2 - 2, 1
    )
    if sp2 <= 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / np.sqrt(sp2))


# ── 1. collect clinical metadata ────────────────────────────────────


def collect_clinical_metadata(base: Path) -> pd.DataFrame:
    """Walk all trials and extract clinical fields into a DataFrame."""
    trials = find_trials(base)
    rows: list[dict] = []
    for name in trials:
        try:
            md = load_trial(str(base), name)["metadata"]
        except Exception:
            continue
        val = md.get("evaluationScoreValue")
        # skip non-numeric scores
        if isinstance(val, str):
            val = None
        vga = md.get("visualGaitAssessment")
        if isinstance(vga, str):
            vga = None
        rows.append(
            {
                "trial": name,
                "subject": md.get("subject"),
                "group": md.get("group"),
                "pathology_key": md.get("pathologyKey"),
                "score_name": md.get("evaluationScoreName"),
                "score_value": float(val) if val is not None else None,
                "deficit_side": md.get("clinicalDeficitSide"),
                "vga": float(vga) if vga is not None else None,
            }
        )
    return pd.DataFrame(rows)


# ── 2. subtype breakdown ────────────────────────────────────────────


def plot_subtype_boxplot(
    df_subj: pd.DataFrame, col: str, title: str, path: Path
) -> None:
    """Boxplot of an asymmetry metric broken down by pathology subtype."""
    order = ["HS", "ACL", "HOA", "KOA", "CIPN", "CVA", "PD", "RIL"]
    present = [pk for pk in order if pk in df_subj["pathology_key"].values]

    fig, ax = plt.subplots(figsize=(12, 6))
    data = [
        df_subj.loc[df_subj["pathology_key"] == pk, col].dropna().values
        for pk in present
    ]
    # color: green for HS, blue for ortho, red for neuro
    color_map = {
        "HS": "#55A868",
        "ACL": "#4C72B0",
        "HOA": "#4C72B0",
        "KOA": "#4C72B0",
        "CIPN": "#C44E52",
        "CVA": "#C44E52",
        "PD": "#C44E52",
        "RIL": "#C44E52",
    }
    bp = ax.boxplot(data, tick_labels=present, patch_artist=True)
    for patch, pk in zip(bp["boxes"], present):
        patch.set_facecolor(color_map.get(pk, "#999999"))
        patch.set_alpha(0.6)
    ax.set_title(title)
    ax.set_ylabel(col)
    ax.set_xlabel("Pathology subtype")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="grey", lw=0.5, ls="--")

    # annotate sample sizes
    for i, pk in enumerate(present):
        n = data[i].size
        ax.text(i + 1, ax.get_ylim()[1] * 0.95, f"n={n}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def subtype_stats(
    df_subj: pd.DataFrame, col: str, ref: str = "HS"
) -> List[Dict]:
    """Run t-tests comparing each subtype against healthy (HS)."""
    ref_vals = df_subj.loc[df_subj["pathology_key"] == ref, col].dropna().values
    results = []
    for pk in sorted(df_subj["pathology_key"].unique()):
        if pk == ref:
            continue
        vals = df_subj.loc[df_subj["pathology_key"] == pk, col].dropna().values
        if vals.size < 2:
            continue
        t, p = sp_stats.ttest_ind(ref_vals, vals, equal_var=False)
        d = _cohens_d(ref_vals, vals)
        results.append(
            {
                "comparison": f"{ref}_vs_{pk}",
                "n_ref": int(ref_vals.size),
                "n_test": int(vals.size),
                "ref_mean": float(np.mean(ref_vals)),
                "test_mean": float(np.mean(vals)),
                "t": float(t),
                "p": float(p),
                "d": d,
            }
        )
    return sorted(results, key=lambda x: abs(x["d"]), reverse=True)


# ── 3. ROC classifier ───────────────────────────────────────────────


def roc_curve_manual(
    labels: np.ndarray, scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve: (fpr, tpr, thresholds).

    labels: 1=positive (healthy), 0=negative (pathological)
    scores: higher = more likely healthy
    """
    thresholds = np.sort(np.unique(scores))[::-1]
    fpr_list, tpr_list = [0.0], [0.0]
    for th in thresholds:
        pred_pos = scores >= th
        tp = np.sum(pred_pos & (labels == 1))
        fp = np.sum(pred_pos & (labels == 0))
        fn = np.sum(~pred_pos & (labels == 1))
        tn = np.sum(~pred_pos & (labels == 0))
        tpr_list.append(tp / max(tp + fn, 1))
        fpr_list.append(fp / max(fp + tn, 1))
    fpr_list.append(1.0)
    tpr_list.append(1.0)
    return np.array(fpr_list), np.array(tpr_list), thresholds


def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Trapezoidal AUC."""
    order = np.argsort(fpr)
    return float(np.trapezoid(tpr[order], fpr[order]))


def optimal_threshold(
    fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray
) -> Tuple[float, float, float]:
    """Youden's J optimal threshold. Returns (threshold, sensitivity, specificity)."""
    # fpr/tpr have one extra element at each end
    j = tpr[1:-1] - fpr[1:-1]
    idx = np.argmax(j)
    return float(thresholds[idx]), float(tpr[idx + 1]), float(1 - fpr[idx + 1])


def plot_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_val: float,
    opt_th: float,
    sens: float,
    spec: float,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.plot(1 - spec, sens, "ro", ms=10, label=f"Opt: th={opt_th:.3f}\nSens={sens:.2f}, Spec={spec:.2f}")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── 4. clinical correlation ─────────────────────────────────────────


def plot_correlation(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
) -> Dict:
    """Scatter plot with Spearman correlation, colored by group."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y, groups = x[mask], y[mask], groups[mask]
    rho, p = sp_stats.spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(10, 7))
    color_map = {"healthy": "#55A868", "ortho": "#4C72B0", "neuro": "#C44E52"}
    for grp in ["healthy", "ortho", "neuro"]:
        m = groups == grp
        if m.sum() > 0:
            ax.scatter(x[m], y[m], c=color_map[grp], label=grp, alpha=0.6, s=40)

    # trend line
    z = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, np.polyval(z, xline), "k--", alpha=0.4, lw=1.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nSpearman rho={rho:.3f}, p={p:.4f} {_sig(p)}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"rho": float(rho), "p": float(p), "n": int(mask.sum())}


# ── main ─────────────────────────────────────────────────────────────


def run(
    base_path: str = "dataset/data",
    out_dir: str = "results",
) -> None:
    base = Path(base_path)
    out = Path(out_dir)
    ensure_dirs(out)
    art = out / "artifacts"
    fig_dir = out / "figures"

    # load existing data
    df_trial = pd.read_csv(art / "asymmetry_per_trial.csv")
    df_subj = pd.read_csv(art / "asymmetry_per_subject.csv")

    print(f"Loaded {len(df_trial)} trials, {len(df_subj)} subjects")

    # ── 1. Subtype breakdown ──
    print("\n" + "=" * 60)
    print("PATHOLOGY SUBTYPE BREAKDOWN (subject-level)")
    print("=" * 60)

    for col in ("stride_absAI", "step_absAI", "stride_abs_diff"):
        plot_subtype_boxplot(
            df_subj,
            col,
            f"Step 07 — {col} by Pathology Subtype",
            fig_dir / f"step07_subtype_{col}.png",
        )
        st = subtype_stats(df_subj, col)
        print(f"\n{col} vs HS:")
        for r in st:
            comp = r["comparison"].replace("HS_vs_", "")
            print(
                f"  {comp:6s}  mean={r['test_mean']:.4f}  "
                f"t={r['t']:+.2f}  p={r['p']:.4f} {_sig(r['p'])}  d={r['d']:+.3f}"
            )

    # save subtype stats
    subtype_results: Dict = {}
    for col in ("stride_absAI", "step_absAI", "stride_abs_diff"):
        subtype_results[col] = subtype_stats(df_subj, col)
    (art / "asymmetry_subtype_stats.json").write_text(
        json.dumps(subtype_results, indent=2, default=str)
    )

    # ── 2. ROC classifier ──
    print("\n" + "=" * 60)
    print("ROC CLASSIFIER (subject-level)")
    print("=" * 60)

    roc_results: Dict = {}
    for col in ("stride_absAI", "step_absAI", "stride_abs_diff"):
        vals = df_subj[col].dropna().values
        grps = df_subj.loc[df_subj[col].notna(), "group"].values
        # label: 1=healthy, 0=pathological
        labels = (grps == "healthy").astype(int)

        fpr, tpr, thresholds = roc_curve_manual(labels, vals)
        auc_val = compute_auc(fpr, tpr)
        th, sens, spec = optimal_threshold(fpr, tpr, thresholds)

        plot_roc(
            fpr, tpr, auc_val, th, sens, spec,
            f"Step 07 — ROC: {col} (Healthy vs Pathological)",
            fig_dir / f"step07_roc_{col}.png",
        )

        roc_results[col] = {
            "auc": auc_val,
            "optimal_threshold": th,
            "sensitivity": sens,
            "specificity": spec,
            "n_healthy": int((labels == 1).sum()),
            "n_pathological": int((labels == 0).sum()),
        }
        print(
            f"  {col}: AUC={auc_val:.3f}  th={th:.4f}  "
            f"sens={sens:.2f}  spec={spec:.2f}"
        )

    (art / "asymmetry_roc.json").write_text(
        json.dumps(roc_results, indent=2)
    )

    # ── 3. Clinical score correlation ──
    print("\n" + "=" * 60)
    print("CLINICAL SCORE CORRELATION")
    print("=" * 60)

    # collect VGA (visual gait assessment) per trial, merge
    clin_df = collect_clinical_metadata(base)
    merged = df_trial.merge(
        clin_df[["trial", "score_value", "vga", "deficit_side"]],
        on="trial",
        how="left",
    )

    corr_results: Dict = {}

    # VGA correlation (available for most trials)
    vga_mask = merged["vga"].notna()
    if vga_mask.sum() > 10:
        for col in ("stride_absAI", "step_absAI"):
            key = f"vga_vs_{col}"
            r = plot_correlation(
                merged.loc[vga_mask, "vga"].values,
                merged.loc[vga_mask, col].values,
                merged.loc[vga_mask, "group"].values,
                "Visual Gait Assessment (0=normal, 4=severe)",
                col,
                f"Step 07 — VGA vs {col}",
                fig_dir / f"step07_corr_vga_{col}.png",
            )
            corr_results[key] = r
            print(f"  VGA vs {col}: rho={r['rho']:.3f}, p={r['p']:.4f} {_sig(r['p'])}, n={r['n']}")

    # Per-pathology clinical score correlation
    for pk in ("CVA", "PD", "RIL", "KOA"):
        sub = merged[(merged["pathology_key"] == pk) & merged["score_value"].notna()]
        if len(sub) < 10:
            continue
        for col in ("stride_absAI",):
            key = f"{pk}_score_vs_{col}"
            score_name = sub.iloc[0].get("pathology_key", pk)
            r = plot_correlation(
                sub["score_value"].values,
                sub[col].values,
                sub["group"].values,
                f"Clinical Score ({pk})",
                col,
                f"Step 07 — {pk} Clinical Score vs {col}",
                fig_dir / f"step07_corr_{pk}_{col}.png",
            )
            corr_results[key] = r
            print(f"  {pk} score vs {col}: rho={r['rho']:.3f}, p={r['p']:.4f} {_sig(r['p'])}, n={r['n']}")

    (art / "asymmetry_correlations.json").write_text(
        json.dumps(corr_results, indent=2, default=str)
    )

    print(f"\nDone. Figures saved to {fig_dir}")


if __name__ == "__main__":
    run()
