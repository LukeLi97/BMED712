import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score


def col_is_feature(c: str) -> bool:
    return c not in {"trial_id", "subject_id", "label", "phase", "win_s", "overlap"}


def cv_with_trial_agg(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if col_is_feature(c)]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    y = df["label"].astype(str)
    groups = df["subject_id"].astype(str)
    trials = df["trial_id"].astype(str)

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    w_bacc: List[float] = []
    w_f1: List[float] = []
    t_bacc: List[float] = []
    t_f1: List[float] = []

    for tr_idx, te_idx in skf.split(X, y, groups):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
        trials_te = trials.iloc[te_idx]
        if len(pd.unique(ytr)) < 2:
            continue
        clf = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(with_mean=True),
            SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True),
        )
        clf.fit(Xtr, ytr)
        ypred = clf.predict(Xte)
        prob = clf.predict_proba(Xte)
        w_bacc.append(balanced_accuracy_score(yte, ypred))
        w_f1.append(f1_score(yte, ypred, average="macro"))

        # trial-level aggregation (mean prob per trial)
        df_fold = pd.DataFrame({
            "trial": trials_te.to_numpy(),
            "y": yte.to_numpy(),
            "pred": ypred,
        })
        # map class order
        classes = clf.classes_.tolist()
        prob_df = pd.DataFrame(prob, columns=[f"p:{c}" for c in classes])
        dfp = pd.concat([df_fold, prob_df], axis=1)
        # aggregate
        g = dfp.groupby("trial")
        meanp = g[[f"p:{c}" for c in classes]].mean()
        true = g["y"].first()
        idx = np.argmax(meanp.to_numpy(), axis=1)
        pred_trial = [classes[i] for i in idx]
        t_bacc.append(balanced_accuracy_score(true, pred_trial))
        t_f1.append(f1_score(true, pred_trial, average="macro"))

    return {
        "window_bacc_mean": float(np.mean(w_bacc) if w_bacc else np.nan),
        "window_f1_mean": float(np.mean(w_f1) if w_f1 else np.nan),
        "trial_bacc_mean": float(np.mean(t_bacc) if t_bacc else np.nan),
        "trial_f1_mean": float(np.mean(t_f1) if t_f1 else np.nan),
    }


def main():
    ap = argparse.ArgumentParser(description="Subject-wise CV with trial-level aggregation")
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    res = cv_with_trial_agg(args.csv)
    print(res)


if __name__ == "__main__":
    main()
