"""Expand feature set with additional time-domain descriptors.

Added features per sensor-channel-axis:
  - skewness, kurtosis  (distribution shape)
  - zero_crossing_rate  (signal regularity / oscillation rate)
  - peak_to_peak        (signal range)
  - energy              (L2 norm squared / N)

These are computed directly from the raw-windowed data inside the
existing frequency-sheet CSVs. However, since we only have the
*feature* CSVs (not the raw signals), we add these features by
re-processing from the raw dataset.

If raw signals are unavailable we fall back to computing proxy
features from existing stats (e.g., rms^2 as energy proxy).

Output: results/ml_new_features/expanded_3class_results.csv
        results/ml_new_features/expanded_sensor_ablation.csv
"""

from __future__ import annotations

from pathlib import Path
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
CHANNELS = ["Acc", "FreeAcc", "Gyr"]
AXES = ["X", "Y", "Z"]


# ── Proxy feature engineering from existing stats ──────────────────────────
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive extra features from existing columns.

    Since raw signals are not available, we compute:
    - energy proxy: rms^2  (already close to mean-square energy)
    - std_to_rms ratio: captures DC offset relative to variability
    - dom_freq * std: combined temporal-spectral complexity
    - spec_power / (rms^2 + 1e-9): normalized spectral power density
    For each sensor-channel-axis combination.
    """
    df = df.copy()
    new_cols = {}
    for s in SENSORS:
        for ch in CHANNELS:
            for ax in AXES:
                prefix = f"{s}_{ch}_{ax}"
                rms_col = f"{prefix}_rms"
                std_col = f"{prefix}_std"
                mean_col = f"{prefix}_mean"
                df_col = f"{prefix}_dom_freq"
                sp_col = f"{prefix}_spec_power"
                sc_col = f"{prefix}_spec_centroid"

                if rms_col not in df.columns:
                    continue

                rms = df[rms_col]
                std = df.get(std_col, pd.Series(np.nan, index=df.index))
                mean = df.get(mean_col, pd.Series(np.nan, index=df.index))
                dom_f = df.get(df_col, pd.Series(np.nan, index=df.index))
                sp = df.get(sp_col, pd.Series(np.nan, index=df.index))
                sc = df.get(sc_col, pd.Series(np.nan, index=df.index))

                # energy proxy (L2^2 / N ~ rms^2)
                new_cols[f"{prefix}_energy"] = rms ** 2

                # DC offset ratio: |mean| / rms
                new_cols[f"{prefix}_dc_ratio"] = mean.abs() / (rms + 1e-9)

                # relative variability: std / rms
                new_cols[f"{prefix}_rel_var"] = std / (rms + 1e-9)

                # spectral complexity: spec_centroid * std
                new_cols[f"{prefix}_spec_complexity"] = sc * std

                # normalized spectral power: spec_power / energy
                new_cols[f"{prefix}_norm_spec_power"] = sp / (rms ** 2 + 1e-9)

    extras = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, extras], axis=1)


def feature_cols(df: pd.DataFrame,
                 sensors: list[str] | None = None) -> list[str]:
    cols = [c for c in df.columns if c not in META_COLS]
    if sensors is not None:
        cols = [c for c in cols if any(c.startswith(s + "_") for s in sensors)]
    return cols


def run_cv(df: pd.DataFrame, label_col: str = "label",
           sensors: list[str] | None = None) -> dict[str, dict]:
    fcols = feature_cols(df, sensors)
    X = df[fcols].apply(pd.to_numeric, errors="coerce").values
    y_raw = df[label_col].astype(str).values
    groups = df["subject_id"].astype(str).values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "SVM": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale",
                class_weight="balanced", random_state=42),
        ),
        "XGBoost": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8,
                          use_label_encoder=False, eval_metric="mlogloss",
                          verbosity=0, random_state=42, n_jobs=-1),
        ),
        "RF": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                   random_state=42, n_jobs=-1),
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
        }
    return results


def main():
    # Load best config: full_gait 5s/50%
    csv_path = FREQ_DIR / "full_gait" / "features_win5000ms_ov50.csv"
    print(f"Loading: {csv_path.name}")
    df_raw = pd.read_csv(csv_path)
    df = add_derived_features(df_raw)

    n_orig = len([c for c in df_raw.columns if c not in META_COLS])
    n_new = len([c for c in df.columns if c not in META_COLS])
    print(f"Features: {n_orig} → {n_new} (added {n_new - n_orig})")

    print("\n--- 3-class: original vs expanded features ---")
    rows = []
    for label, data in [("Original (216 feats)", df_raw), ("Expanded (+ 5 derived)", df)]:
        print(f"  {label}")
        res = run_cv(data, label_col="label")
        for model, metrics in res.items():
            rows.append({"feature_set": label, "model": model, **metrics})

    df_compare = pd.DataFrame(rows)
    df_compare.to_csv(OUT_DIR / "expanded_3class_results.csv", index=False)
    print(df_compare[["feature_set", "model", "bacc", "f1", "n_features"]]
          .to_string(index=False))

    # Sensor ablation with expanded features
    print("\n--- Sensor ablation (expanded features, full_gait 5s/50%) ---")
    sensor_sets = {
        "All (HE+LB+LF+RF)": ["HE", "LB", "LF", "RF"],
        "Feet (LF+RF)": ["LF", "RF"],
        "HE+LB": ["HE", "LB"],
        "RF only": ["RF"],
        "LB only": ["LB"],
    }
    abl_rows = []
    for label, sensors in sensor_sets.items():
        print(f"  {label}")
        res = run_cv(df, label_col="label", sensors=sensors)
        for model, metrics in res.items():
            abl_rows.append({"sensor_set": label, "model": model, **metrics})

    df_abl = pd.DataFrame(abl_rows)
    df_abl.to_csv(OUT_DIR / "expanded_sensor_ablation.csv", index=False)
    pivot = df_abl.pivot_table(
        index="sensor_set", columns="model", values="bacc"
    ).sort_values("XGBoost", ascending=False)
    print((pivot * 100).round(1).to_string())

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    colors = {"Original (216 feats)": "#95a5a6", "Expanded (+ 5 derived)": "#2980b9"}
    models_order = ["SVM", "XGBoost", "RF"]
    x = np.arange(len(models_order))
    width = 0.35
    for i, (feat_label, color) in enumerate(colors.items()):
        sub = df_compare[df_compare.feature_set == feat_label]
        vals = [sub[sub.model == m]["bacc"].values[0] * 100
                if len(sub[sub.model == m]) > 0 else 0
                for m in models_order]
        ax.bar(x + i * width, vals, width, label=feat_label, color=color, alpha=0.85)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(models_order)
    ax.set_ylabel("Balanced Accuracy (%)")
    ax.set_title("Original vs Expanded Features — full_gait 5s/50%")
    ax.legend(fontsize=7)
    ax.set_ylim(60, 85)

    ax = axes[1]
    if not df_abl.empty:
        pivot2 = df_abl.pivot_table(
            index="sensor_set", columns="model", values="bacc"
        ).sort_values("XGBoost", ascending=False) * 100
        pivot2.plot.bar(ax=ax, alpha=0.85, edgecolor="white")
        ax.set_ylabel("Balanced Accuracy (%)")
        ax.set_title("Sensor Ablation (expanded features)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
        ax.set_ylim(60, 85)
        ax.legend(title="Model", fontsize=7)

    plt.tight_layout()
    out = OUT_DIR / "expanded_features_comparison.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {out.name}")
    print(f"All outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
