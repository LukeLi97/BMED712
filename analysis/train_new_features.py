"""Train 3-class and 8-class classifiers on new frequency-sheet features.

Experiments:
  A) 3-class (healthy / neuro / ortho)  — full_gait best configs
  B) 8-class (cohort-level: HS/RIL/PD/CVA/CIPN/KOA/HOA/ACL)
  C) Sensor ablation on best config (3-class, full_gait 5s/50%)
  D) Phase comparison (pre/full/post/uturn best each)

Models: SVM, XGBoost, Random Forest
CV: 5-fold StratifiedGroupKFold (grouped by subject_id)
Metric: balanced accuracy + macro-F1

Outputs: results/ml_new_features/
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from xgboost import XGBClassifier

FREQ_DIR = Path(__file__).resolve().parents[2] / "frequency sheets"
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "ml_new_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_COLS = {"subject_id", "trial_id", "window_idx", "label", "cohort",
             "phase", "win_s", "overlap"}

SENSORS = ["HE", "LB", "LF", "RF"]
SENSOR_SETS = {
    "All (HE+LB+LF+RF)": ["HE", "LB", "LF", "RF"],
    "Feet (LF+RF)": ["LF", "RF"],
    "LF only": ["LF"],
    "RF only": ["RF"],
    "LB only": ["LB"],
    "HE only": ["HE"],
    "LB+RF": ["LB", "RF"],
    "HE+LB": ["HE", "LB"],
}


def load_csv(phase: str, win_s: float, overlap: int) -> pd.DataFrame:
    fname = f"features_win{int(win_s * 1000)}ms_ov{overlap}.csv"
    path = FREQ_DIR / phase / fname
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def feature_cols(df: pd.DataFrame, sensors: Sequence[str] | None = None) -> list[str]:
    cols = [c for c in df.columns if c not in META_COLS]
    if sensors is not None:
        cols = [c for c in cols if any(c.startswith(s + "_") for s in sensors)]
    return cols


def run_cv(df: pd.DataFrame, label_col: str = "label",
           sensors: Sequence[str] | None = None,
           n_splits: int = 5) -> dict[str, dict]:
    """Run 5-fold subject-grouped CV for SVM, XGB, RF.

    Returns dict[model_name] -> {bacc, f1}
    """
    fcols = feature_cols(df, sensors)
    X = df[fcols].apply(pd.to_numeric, errors="coerce").values
    y_raw = df[label_col].astype(str).values
    groups = df["subject_id"].astype(str).values

    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    models = {
        "SVM": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale",
                class_weight="balanced", random_state=42),
        ),
        "XGBoost": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="mlogloss",
                verbosity=0, random_state=42,
                n_jobs=-1,
            ),
        ),
        "RF": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=200, max_depth=None,
                class_weight="balanced", random_state=42, n_jobs=-1,
            ),
        ),
    }

    results = {}
    for name, clf in models.items():
        baccs, f1s = [], []
        for tr_idx, te_idx in skf.split(X, y, groups):
            Xtr, Xte = X[tr_idx], X[te_idx]
            ytr, yte = y[tr_idx], y[te_idx]
            if len(np.unique(ytr)) < 2:
                continue
            clf.fit(Xtr, ytr)
            ypred = clf.predict(Xte)
            baccs.append(balanced_accuracy_score(yte, ypred))
            f1s.append(f1_score(yte, ypred, average="macro"))
        results[name] = {
            "bacc": float(np.mean(baccs)) if baccs else np.nan,
            "f1": float(np.mean(f1s)) if f1s else np.nan,
            "n_features": len(fcols),
            "n_windows": len(df),
            "n_classes": n_classes,
        }
    return results


def experiment_a_3class() -> pd.DataFrame:
    """3-class comparison across phases and best window configs."""
    print("\n=== A) 3-CLASS EXPERIMENTS ===")
    configs = [
        ("full_gait", 5.0, 50),
        ("full_gait", 5.0, 25),
        ("full_gait", 3.0, 50),
        ("pre_uturn", 5.0, 50),
        ("pre_uturn", 6.0, 50),
        ("post_uturn", 6.0, 50),
        ("post_uturn", 5.0, 50),
        ("uturn", 1.0, 50),
        ("uturn", 1.28, 50),
    ]
    rows = []
    for phase, win, ov in configs:
        df = load_csv(phase, win, ov)
        if df.empty:
            continue
        print(f"  {phase} {win}s/{ov}%  n={len(df)}")
        res = run_cv(df, label_col="label")
        for model, metrics in res.items():
            rows.append({
                "phase": phase, "win_s": win, "overlap_pct": ov,
                "model": model,
                **metrics,
            })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / "3class_results.csv", index=False)
    print(df_out[["phase", "win_s", "overlap_pct", "model", "bacc", "f1"]]
          .sort_values(["phase", "model", "bacc"], ascending=[True, True, False])
          .to_string(index=False))
    return df_out


def experiment_b_8class() -> pd.DataFrame:
    """8-class (cohort-level) on full_gait best config."""
    print("\n=== B) 8-CLASS EXPERIMENTS ===")
    configs = [
        ("full_gait", 5.0, 50),
        ("pre_uturn", 5.0, 50),
        ("post_uturn", 6.0, 50),
        ("uturn", 1.0, 50),
    ]
    rows = []
    for phase, win, ov in configs:
        df = load_csv(phase, win, ov)
        if df.empty:
            continue
        print(f"  {phase} {win}s/{ov}%  n={len(df)}")
        res = run_cv(df, label_col="cohort")
        for model, metrics in res.items():
            rows.append({
                "phase": phase, "win_s": win, "overlap_pct": ov,
                "model": model, **metrics,
            })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / "8class_results.csv", index=False)
    print(df_out[["phase", "win_s", "overlap_pct", "model", "bacc", "f1"]]
          .to_string(index=False))
    return df_out


def experiment_c_sensor_ablation() -> pd.DataFrame:
    """Sensor ablation on full_gait 5s/50% (3-class)."""
    print("\n=== C) SENSOR ABLATION (full_gait 5s/50%, 3-class) ===")
    df = load_csv("full_gait", 5.0, 50)
    if df.empty:
        return pd.DataFrame()

    rows = []
    for sensor_label, sensors in SENSOR_SETS.items():
        fcols = feature_cols(df, sensors)
        print(f"  {sensor_label} ({len(fcols)} features)")
        res = run_cv(df, label_col="label", sensors=sensors)
        for model, metrics in res.items():
            rows.append({
                "sensor_set": sensor_label,
                "n_sensors": len(sensors),
                "model": model,
                **metrics,
            })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / "sensor_ablation_results.csv", index=False)
    print(df_out[["sensor_set", "model", "bacc", "f1", "n_features"]]
          .sort_values(["model", "bacc"], ascending=[True, False])
          .to_string(index=False))
    return df_out


def plot_results(df_3class: pd.DataFrame, df_ablation: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot A: 3-class BAcc by phase ---
    ax = axes[0]
    best_per_phase = (
        df_3class.groupby(["phase", "model"])["bacc"].max()
        .reset_index()
    )
    phases = sorted(best_per_phase.phase.unique())
    models = ["SVM", "XGBoost", "RF"]
    colors = {"SVM": "#3498DB", "XGBoost": "#E74C3C", "RF": "#2ECC71"}
    x = np.arange(len(phases))
    width = 0.25
    for i, model in enumerate(models):
        sub = best_per_phase[best_per_phase.model == model]
        vals = [sub[sub.phase == p]["bacc"].values[0]
                if len(sub[sub.phase == p]) > 0 else np.nan
                for p in phases]
        ax.bar(x + i * width, [v * 100 if not np.isnan(v) else 0 for v in vals],
               width, label=model, color=colors[model], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels([p.replace("_", " ") for p in phases], rotation=20, ha="right")
    ax.set_ylabel("Balanced Accuracy (%)")
    ax.set_title("3-Class Balanced Accuracy by Phase (best window config)")
    ax.legend()
    ax.set_ylim(0, 100)
    ax.axhline(33.3, color="grey", linestyle="--", linewidth=0.8, label="Chance")

    # --- Plot B: Sensor ablation ---
    ax = axes[1]
    if not df_ablation.empty:
        pivot = df_ablation.pivot_table(
            index="sensor_set", columns="model", values="bacc", aggfunc="mean"
        )
        pivot = pivot * 100
        pivot.plot.bar(ax=ax, color=[colors[m] for m in pivot.columns],
                       alpha=0.85, edgecolor="white")
        ax.set_ylabel("Balanced Accuracy (%)")
        ax.set_title("Sensor Ablation — full_gait 5s/50% (3-class)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0, 100)
        ax.axhline(33.3, color="grey", linestyle="--", linewidth=0.8)
        ax.legend(title="Model")

    plt.tight_layout()
    out = OUT_DIR / "ml_results_summary.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  Summary plot saved: {out.name}")


def main():
    df_3class = experiment_a_3class()
    df_8class = experiment_b_8class()
    df_ablation = experiment_c_sensor_ablation()

    plot_results(df_3class, df_ablation)

    print(f"\nAll results saved to: {OUT_DIR}")
    print("\n=== SUMMARY ===")
    if not df_3class.empty:
        best = df_3class.loc[df_3class["bacc"].idxmax()]
        print(f"Best 3-class: {best['model']} on {best['phase']} "
              f"{best['win_s']}s/{best['overlap_pct']}%  "
              f"BAcc={best['bacc']*100:.1f}%  F1={best['f1']*100:.1f}%")
    if not df_8class.empty:
        best8 = df_8class.loc[df_8class["bacc"].idxmax()]
        print(f"Best 8-class: {best8['model']} on {best8['phase']} "
              f"{best8['win_s']}s/{best8['overlap_pct']}%  "
              f"BAcc={best8['bacc']*100:.1f}%  F1={best8['f1']*100:.1f}%")


if __name__ == "__main__":
    main()
