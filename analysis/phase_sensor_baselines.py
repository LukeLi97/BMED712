import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


PHASE2CSV = {
    "pre_uturn": Path("results/features_pre_uturn.csv"),
    "uturn": Path("results/features_uturn.csv"),
    "post_uturn": Path("results/features_post_uturn.csv"),
    "gait_full": Path("results/features_gait_full.csv"),
}


def col_is_feature(c: str) -> bool:
    return c not in {"trial_id", "subject_id", "label"}


def subset_sensor(df: pd.DataFrame, sensor: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith(sensor + "_") and col_is_feature(c)]
    # keep for safety if present at end
    sub = df[cols].copy()
    # ensure numeric dtype
    for c in sub.columns:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    return sub


def macro_ovr_auc(y_true: np.ndarray, prob: np.ndarray, classes: List[str]) -> float:
    # y_true are label strings; need to binarize to one-vs-rest
    y_idx, cls = pd.factorize(y_true)
    # probabilities shape [n_samples, n_classes]
    try:
        y_bin = label_binarize(y_idx, classes=list(range(len(cls))))
        return float(roc_auc_score(y_bin, prob, average="macro", multi_class="ovr"))
    except Exception:
        return float("nan")


def eval_models(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> Tuple[str, Dict[str, float]]:
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    classes = sorted(y.unique().tolist())
    models = {
        "LR": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(with_mean=True),
            LogisticRegression(max_iter=400, n_jobs=1, class_weight="balanced"),
        ),
        "SVM": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(with_mean=True),
            SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True),
        ),
        "RF": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample"),
        ),
    }

    agg: Dict[str, Dict[str, List[float]]] = {k: {m: [] for m in ["acc", "bacc", "f1", "auc"]} for k in models}

    n_folds_used = 0
    for tr, te in skf.split(X, y, groups):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        # Guard: some subject-wise splits may collapse to a single class in train
        if len(pd.unique(ytr)) < 2:
            continue
        for name, clf in models.items():
            clf.fit(Xtr, ytr)
            pred = clf.predict(Xte)
            acc = accuracy_score(yte, pred)
            bacc = balanced_accuracy_score(yte, pred)
            f1 = f1_score(yte, pred, average="macro")
            # AUC (macro OvR), needs predict_proba or decision_function mapped to probs
            prob = None
            if hasattr(clf, "predict_proba"):
                try:
                    prob = clf.predict_proba(Xte)
                except Exception:
                    prob = None
            if prob is None and hasattr(clf, "decision_function"):
                try:
                    dec = clf.decision_function(Xte)
                    # map decision scores to pseudo-prob with softmax for multi-class
                    from scipy.special import softmax  # type: ignore

                    if dec.ndim == 1:
                        prob = np.vstack([1 - 1 / (1 + np.exp(dec)), 1 / (1 + np.exp(dec))]).T
                    else:
                        prob = softmax(dec, axis=1)
                except Exception:
                    prob = None
            auc = macro_ovr_auc(yte.to_numpy(), prob, classes) if prob is not None else float("nan")
            agg[name]["acc"].append(acc)
            agg[name]["bacc"].append(bacc)
            agg[name]["f1"].append(f1)
            agg[name]["auc"].append(auc)
        n_folds_used += 1

    if n_folds_used == 0:
        # No valid folds; return a neutral result
        return "LR", {
            "acc_mean": float("nan"),
            "acc_std": float("nan"),
            "bacc_mean": float("nan"),
            "bacc_std": float("nan"),
            "macro_f1_mean": float("nan"),
            "macro_f1_std": float("nan"),
            "auc_macro_ovr_mean": float("nan"),
            "auc_macro_ovr_std": float("nan"),
        }

    # choose best model by balanced accuracy mean
    best_name = None
    best_bacc = -1.0
    stats: Dict[str, float] = {}
    for name, m in agg.items():
        acc_mean = float(np.mean(m["acc"]))
        bacc_mean = float(np.mean(m["bacc"]))
        f1_mean = float(np.mean(m["f1"]))
        auc_mean = float(np.nanmean(m["auc"]))
        acc_std = float(np.std(m["acc"]))
        bacc_std = float(np.std(m["bacc"]))
        f1_std = float(np.std(m["f1"]))
        auc_std = float(np.nanstd(m["auc"]))
        stats[f"{name}_acc_mean"] = acc_mean
        stats[f"{name}_acc_std"] = acc_std
        stats[f"{name}_bacc_mean"] = bacc_mean
        stats[f"{name}_bacc_std"] = bacc_std
        stats[f"{name}_f1_mean"] = f1_mean
        stats[f"{name}_f1_std"] = f1_std
        stats[f"{name}_auc_mean"] = auc_mean
        stats[f"{name}_auc_std"] = auc_std
        if bacc_mean > best_bacc:
            best_bacc = bacc_mean
            best_name = name

    # flatten best model summary for CSV
    best_prefix = best_name if best_name is not None else "LR"
    out = {
        "model": best_prefix,
        "acc_mean": stats.get(f"{best_prefix}_acc_mean", float("nan")),
        "acc_std": stats.get(f"{best_prefix}_acc_std", float("nan")),
        "bacc_mean": stats.get(f"{best_prefix}_bacc_mean", float("nan")),
        "bacc_std": stats.get(f"{best_prefix}_bacc_std", float("nan")),
        "macro_f1_mean": stats.get(f"{best_prefix}_f1_mean", float("nan")),
        "macro_f1_std": stats.get(f"{best_prefix}_f1_std", float("nan")),
        "auc_macro_ovr_mean": stats.get(f"{best_prefix}_auc_mean", float("nan")),
        "auc_macro_ovr_std": stats.get(f"{best_prefix}_auc_std", float("nan")),
    }
    # also keep per-model metrics to allow deeper inspection
    out_all = {f"{k}": v for k, v in stats.items()}
    out.update(out_all)
    return best_prefix, out


def main():
    ap = argparse.ArgumentParser(description="Per-phase single-sensor baselines and summary merge")
    ap.add_argument("--summary", default=Path("results/sensor_phase_summary.csv"))
    ap.add_argument("--out", default=Path("results/sensor_phase_summary.csv"))
    args = ap.parse_args()

    if not Path(args.summary).exists():
        raise SystemExit("Missing summary CSV; run dataset.quick_start.summarize_sensors first.")
    df_sum = pd.read_csv(args.summary)
    rows = []
    for phase, path in PHASE2CSV.items():
        if not path.exists():
            continue
        sub = df_sum[df_sum["phase"] == phase]
        if sub.empty:
            continue
        # choose best-by-effect-size sensor within the phase
        row_best = sub.sort_values("best_score", ascending=False).iloc[0]
        sensor = str(row_best["sensor"]).strip()
        df = pd.read_csv(path)
        X = subset_sensor(df, sensor)
        if X.empty:
            continue
        y = df["label"].astype(str)
        groups = df["subject_id"].astype(str)
        model_name, stats = eval_models(X, y, groups)
        merged = row_best.to_dict()
        merged.update(stats)
        rows.append(merged)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
