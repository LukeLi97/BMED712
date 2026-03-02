"""Train baseline ML models with and without temporal asymmetry features.

Merges the sensor-channel features (from train_baseline) with the
asymmetry metrics (from asymmetry.py) and compares 5-fold CV performance
across four configurations:

  1. sensor-only        — original time/freq features from all 4 IMUs
  2. asymmetry-only     — stride/step AI, abs-diff, CV, etc.
  3. sensor+asymmetry   — combined feature set
  4. feet+asymmetry     — LF+RF sensor features + asymmetry (minimal setup)

Usage:
    python analysis/train_with_asymmetry.py
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

try:
    from xgboost import XGBClassifier  # type: ignore

    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.pipeline import ensure_dirs, find_trials
from analysis.train_baseline import collect_features, select_feature_columns

# ── asymmetry feature columns to use ─────────────────────────────────

# These are the most discriminative metrics from the asymmetry analysis:
#   stride_absAI  (d=0.77)  — strongest single discriminator
#   step_absAI    (d=0.42)
#   stride_abs_diff (d=0.70) — absolute L-R stride time difference
#   step_abs_diff  (d=0.38)
#   step_CV_l     (d=0.55)  — within-trial gait variability
#   step_CV_r
#   stride_CV_l
#   stride_CV_r
#   stride_AI     — signed, captures directional bias
#   step_AI
#   mean_step_time — walking speed (not significant, but useful covariate)

ASYM_COLS = [
    "stride_absAI",
    "step_absAI",
    "stride_abs_diff",
    "step_abs_diff",
    "stride_AI",
    "step_AI",
    "step_CV_l",
    "step_CV_r",
    "stride_CV_l",
    "stride_CV_r",
    "mean_step_time",
]


# ── CV experiment (mirrors train_baseline.run_cv_experiment) ─────────


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    subj: pd.Series,
    tag: str,
    out_dir: Path,
    n_splits: int = 5,
) -> Dict:
    """Run StratifiedGroupKFold CV with LR/RF/SVM(/XGB) and save results."""
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_enc, classes = pd.factorize(y)
    groups = subj.to_numpy()

    models = {
        "lr": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=300, class_weight="balanced", multi_class="auto"
            ),
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
        ),
        "svm": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),
        ),
    }
    if _HAS_XGB:
        models["xgb"] = XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective=(
                "multi:softprob" if len(classes) > 2 else "binary:logistic"
            ),
            num_class=int(len(classes)) if len(classes) > 2 else None,
            tree_method="hist",
            eval_metric="mlogloss" if len(classes) > 2 else "logloss",
            random_state=42,
            n_jobs=0,
        )

    fold_metrics: Dict[str, List[Dict]] = {k: [] for k in models}
    cms: Dict[str, np.ndarray] = {
        k: np.zeros((len(classes), len(classes)), dtype=int) for k in models
    }

    for train_idx, test_idx in skf.split(X, y_enc, groups):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y_enc[train_idx], y_enc[test_idx]

        for name, clf in models.items():
            Xtr_in = Xtr.to_numpy() if name == "xgb" else Xtr
            Xte_in = Xte.to_numpy() if name == "xgb" else Xte
            clf.fit(Xtr_in, ytr)
            pred = clf.predict(Xte_in)
            fold_metrics[name].append(
                {
                    "macro_f1": float(f1_score(yte, pred, average="macro")),
                    "balanced_acc": float(balanced_accuracy_score(yte, pred)),
                }
            )
            cms[name] += confusion_matrix(
                yte, pred, labels=list(range(len(classes)))
            )

    # Build summary
    summary: Dict = {"tag": tag, "n_features": int(X.shape[1]), "classes": list(map(str, classes))}
    for name in models:
        f1s = [m["macro_f1"] for m in fold_metrics[name]]
        bas = [m["balanced_acc"] for m in fold_metrics[name]]
        summary[name] = {
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
            "balanced_acc_mean": float(np.mean(bas)),
            "balanced_acc_std": float(np.std(bas)),
        }

    # Save JSON
    art = out_dir / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    (art / f"metrics_{tag}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Plot confusion matrices
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ncols = len(models)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    if ncols == 1:
        axes = [axes]
    for ax, name in zip(axes, models):
        ConfusionMatrixDisplay(cms[name], display_labels=classes).plot(
            ax=ax, colorbar=False
        )
        ax.set_title(f"{name.upper()} — {tag}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"step08_confusion_{tag}.png", dpi=150)
    plt.close(fig)

    return summary


# ── feature importance analysis ──────────────────────────────────────


def plot_importance_combined(
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: Path,
    tag: str,
    topk: int = 25,
) -> List[Tuple[str, float]]:
    """Train RF on full data and plot top-k feature importances."""
    y_enc, _ = pd.factorize(y)
    rf = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced_subsample"
    )
    rf.fit(X, y_enc)
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:topk]
    names = [X.columns[i] for i in idx]
    vals = importances[idx]

    # Color asymmetry features differently
    colors = []
    for n in names:
        if any(n.startswith(ac) or n == ac for ac in ASYM_COLS):
            colors.append("#C44E52")  # red for asymmetry
        else:
            colors.append("#4C72B0")  # blue for sensor

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(vals)), vals[::-1], color=colors[::-1])
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title(f"Step 08 — RF Feature Importance ({tag}, top-{topk})")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#C44E52", label="Asymmetry features"),
        Patch(facecolor="#4C72B0", label="Sensor features"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()
    fig.savefig(
        out_dir / "figures" / f"step08_importance_{tag}.png", dpi=150
    )
    plt.close(fig)
    return list(zip(names, vals.tolist()))


# ── comparison bar chart ─────────────────────────────────────────────


def plot_comparison(
    results: Dict[str, Dict],
    metric: str,
    title: str,
    path: Path,
) -> None:
    """Grouped bar chart comparing configurations across models."""
    configs = list(results.keys())
    models_avail = [k for k in ("lr", "rf", "svm", "xgb") if k in results[configs[0]]]

    x = np.arange(len(configs))
    width = 0.8 / len(models_avail)
    colors = {"lr": "#55A868", "rf": "#4C72B0", "svm": "#C44E52", "xgb": "#8172B2"}

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models_avail):
        means = [results[c][model][f"{metric}_mean"] for c in configs]
        stds = [results[c][model][f"{metric}_std"] for c in configs]
        offset = (i - len(models_avail) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=model.upper(),
            color=colors.get(model, "#999999"),
            alpha=0.8,
            capsize=3,
        )
        # Annotate values
        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{m:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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

    # ── 1. Load sensor features ──
    print("=" * 60)
    print("STEP 08 — ML BASELINE WITH ASYMMETRY FEATURES")
    print("=" * 60)

    trials = find_trials(base, limit=None)
    X_sensor, y, subj, tr_ids, pkey = collect_features(base, trials)
    print(f"Sensor features: {X_sensor.shape[1]} features, {len(y)} trials")

    # ── 2. Load asymmetry features ──
    asym_csv = art / "asymmetry_per_trial.csv"
    if not asym_csv.exists():
        print(
            f"ERROR: {asym_csv} not found. Run analysis/asymmetry.py first."
        )
        return
    df_asym = pd.read_csv(asym_csv)
    print(f"Asymmetry CSV: {len(df_asym)} trials, {len(df_asym.columns)} columns")

    # ── 3. Merge by trial name ──
    # Build a trial-indexed DataFrame for asymmetry features
    asym_available = [c for c in ASYM_COLS if c in df_asym.columns]
    print(f"Using {len(asym_available)} asymmetry features: {asym_available}")

    # Create index mapping: trial name → row in X_sensor
    trial_to_idx = {name: i for i, name in enumerate(tr_ids)}
    asym_trial_set = set(df_asym["trial"].values)

    # Build asymmetry feature matrix aligned with X_sensor
    asym_rows = []
    matched = 0
    for tr_name in tr_ids:
        if tr_name in asym_trial_set:
            row = df_asym.loc[df_asym["trial"] == tr_name, asym_available]
            if len(row) == 1:
                asym_rows.append(row.iloc[0].to_dict())
                matched += 1
            else:
                asym_rows.append({c: np.nan for c in asym_available})
        else:
            asym_rows.append({c: np.nan for c in asym_available})

    X_asym = pd.DataFrame(asym_rows, index=X_sensor.index)
    X_asym = X_asym.fillna(0.0)
    print(f"Matched {matched}/{len(tr_ids)} trials with asymmetry data")

    # ── 4. Build feature configurations ──
    X_combined = pd.concat([X_sensor, X_asym], axis=1)

    # Feet-only sensor features + asymmetry
    feet_cols = select_feature_columns(X_sensor, ["LF", "RF"])
    X_feet = X_sensor[feet_cols]
    X_feet_asym = pd.concat([X_feet, X_asym], axis=1)

    configs = {
        "sensor_only": X_sensor,
        "asym_only": X_asym,
        "sensor+asym": X_combined,
        "feet+asym": X_feet_asym,
    }

    # ── 5. Run CV for each configuration ──
    results: Dict[str, Dict] = {}
    for config_name, X_cfg in configs.items():
        print(f"\n{'─' * 50}")
        print(f"Config: {config_name} ({X_cfg.shape[1]} features)")
        print("─" * 50)
        summary = run_cv(X_cfg, y, subj, f"3class_{config_name}", out)
        results[config_name] = summary

        # Print results
        for model in ("lr", "rf", "svm", "xgb"):
            if model not in summary:
                continue
            m = summary[model]
            print(
                f"  {model.upper():4s}  "
                f"F1={m['macro_f1_mean']:.3f}±{m['macro_f1_std']:.3f}  "
                f"BAcc={m['balanced_acc_mean']:.3f}±{m['balanced_acc_std']:.3f}"
            )

    # ── 5b. Matched-only experiments (fair comparison) ──
    # Only use the 974 trials that have valid asymmetry data
    match_mask = pd.Series([tr in asym_trial_set for tr in tr_ids])
    if match_mask.sum() > 0:
        print(f"\n{'=' * 60}")
        print(f"MATCHED-ONLY SUBSET ({match_mask.sum()} trials)")
        print("=" * 60)
        X_m_sensor = X_sensor.loc[match_mask.values].reset_index(drop=True)
        X_m_asym = X_asym.loc[match_mask.values].reset_index(drop=True)
        X_m_combined = X_combined.loc[match_mask.values].reset_index(drop=True)
        y_m = y[match_mask.values].reset_index(drop=True)
        subj_m = subj[match_mask.values].reset_index(drop=True)

        matched_configs = {
            "matched_sensor": X_m_sensor,
            "matched_sensor+asym": X_m_combined,
            "matched_asym_only": X_m_asym,
        }
        for config_name, X_cfg in matched_configs.items():
            print(f"\n{'─' * 50}")
            print(f"Config: {config_name} ({X_cfg.shape[1]} features, {len(y_m)} trials)")
            print("─" * 50)
            summary = run_cv(X_cfg, y_m, subj_m, f"3class_{config_name}", out)
            results[config_name] = summary
            for model in ("lr", "rf", "svm", "xgb"):
                if model not in summary:
                    continue
                m = summary[model]
                print(
                    f"  {model.upper():4s}  "
                    f"F1={m['macro_f1_mean']:.3f}±{m['macro_f1_std']:.3f}  "
                    f"BAcc={m['balanced_acc_mean']:.3f}±{m['balanced_acc_std']:.3f}"
                )

    # ── 6. Comparison plots ──
    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print("=" * 60)

    plot_comparison(
        results,
        "macro_f1",
        "Step 08 — Macro-F1: Sensor vs Asymmetry Features (3-class)",
        fig_dir / "step08_comparison_f1.png",
    )
    plot_comparison(
        results,
        "balanced_acc",
        "Step 08 — Balanced Accuracy: Sensor vs Asymmetry Features (3-class)",
        fig_dir / "step08_comparison_bacc.png",
    )

    # Matched-only comparison plots
    matched_results = {k: v for k, v in results.items() if k.startswith("matched_")}
    if matched_results:
        plot_comparison(
            matched_results,
            "macro_f1",
            "Step 08 — Macro-F1: Matched Trials Only (974 trials, fair comparison)",
            fig_dir / "step08_matched_comparison_f1.png",
        )
        plot_comparison(
            matched_results,
            "balanced_acc",
            "Step 08 — BAcc: Matched Trials Only (974 trials, fair comparison)",
            fig_dir / "step08_matched_comparison_bacc.png",
        )

    # ── 7. Feature importance for combined model ──
    top_feats = plot_importance_combined(
        X_combined, y, out, "sensor_plus_asym", topk=25
    )
    print("\nTop-10 features (sensor+asymmetry RF):")
    for name, imp in top_feats[:10]:
        tag = " ★ ASYM" if name in ASYM_COLS else ""
        print(f"  {name:40s}  {imp:.4f}{tag}")

    # Count asymmetry features in top-25
    asym_in_top = sum(1 for n, _ in top_feats if n in ASYM_COLS)
    print(f"\nAsymmetry features in top-25: {asym_in_top}/{len(asym_available)}")

    # ── 8. Compute deltas ──
    print(f"\n{'=' * 60}")
    print("IMPROVEMENT SUMMARY (all trials)")
    print("=" * 60)

    deltas: Dict[str, Dict] = {}
    baseline = results["sensor_only"]
    for config_name, summary in results.items():
        if config_name == "sensor_only" or config_name.startswith("matched_"):
            continue
        d: Dict[str, Dict] = {}
        for model in ("lr", "rf", "svm", "xgb"):
            if model not in summary or model not in baseline:
                continue
            d[model] = {
                "delta_f1": summary[model]["macro_f1_mean"]
                - baseline[model]["macro_f1_mean"],
                "delta_bacc": summary[model]["balanced_acc_mean"]
                - baseline[model]["balanced_acc_mean"],
            }
        deltas[config_name] = d

    for config_name, d in deltas.items():
        print(f"\n  {config_name} vs sensor_only:")
        for model, vals in d.items():
            sign_f1 = "+" if vals["delta_f1"] >= 0 else ""
            sign_ba = "+" if vals["delta_bacc"] >= 0 else ""
            print(
                f"    {model.upper():4s}  "
                f"ΔF1={sign_f1}{vals['delta_f1']:.3f}  "
                f"ΔBAcc={sign_ba}{vals['delta_bacc']:.3f}"
            )

    # Matched-only deltas (fair comparison on same trial subset)
    if "matched_sensor" in results and "matched_sensor+asym" in results:
        print(f"\n{'=' * 60}")
        print("IMPROVEMENT SUMMARY (matched trials only — fair comparison)")
        print("=" * 60)
        base_m = results["matched_sensor"]
        for config_name in ("matched_sensor+asym", "matched_asym_only"):
            if config_name not in results:
                continue
            summary = results[config_name]
            print(f"\n  {config_name} vs matched_sensor:")
            for model in ("lr", "rf", "svm", "xgb"):
                if model not in summary or model not in base_m:
                    continue
                df1 = summary[model]["macro_f1_mean"] - base_m[model]["macro_f1_mean"]
                dba = summary[model]["balanced_acc_mean"] - base_m[model]["balanced_acc_mean"]
                sf1 = "+" if df1 >= 0 else ""
                sba = "+" if dba >= 0 else ""
                print(f"    {model.upper():4s}  ΔF1={sf1}{df1:.3f}  ΔBAcc={sba}{dba:.3f}")
                deltas[config_name] = deltas.get(config_name, {})
                deltas[config_name][model] = {"delta_f1": df1, "delta_bacc": dba}

    # ── 9. Save consolidated results ──
    consolidated = {
        "experiment": "asymmetry_feature_integration",
        "n_trials": int(len(y)),
        "n_subjects": int(subj.nunique()),
        "n_sensor_features": int(X_sensor.shape[1]),
        "n_asymmetry_features": len(asym_available),
        "asymmetry_features_used": asym_available,
        "matched_trials": matched,
        "configs": {},
        "deltas_vs_sensor_only": deltas,
    }
    for config_name, summary in results.items():
        consolidated["configs"][config_name] = {
            k: v for k, v in summary.items() if k not in ("tag", "classes")
        }

    (art / "asymmetry_ml_integration.json").write_text(
        json.dumps(consolidated, indent=2), encoding="utf-8"
    )

    print(f"\nDone. Results saved to {art / 'asymmetry_ml_integration.json'}")
    print(f"Figures: {fig_dir / 'step08_*.png'}")


if __name__ == "__main__":
    run()
