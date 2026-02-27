import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score


FEAT_EXCLUDE = {"trial_id", "subject_id", "label", "phase", "win_s", "overlap"}


def col_is_feature(c: str) -> bool:
    if c in FEAT_EXCLUDE:
        return False
    if c.startswith("p:"):
        return False
    return True


def load_phase_table(path: Path, phase: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["phase"] = phase
    return df


def run_cv(phase2csv: Dict[str, str]) -> dict:
    # Load and concatenate
    frames = []
    for ph, p in phase2csv.items():
        frames.append(load_phase_table(Path(p), ph))
    data = pd.concat(frames, ignore_index=True)
    # ensure string dtypes
    data["label"] = data["label"].astype(str)
    data["subject_id"] = data["subject_id"].astype(str)
    data["trial_id"] = data["trial_id"].astype(str)
    # CV setup on subjects
    y = data["label"].astype(str)
    groups = data["subject_id"].astype(str)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    t_bacc = []
    t_f1 = []

    phases = list(phase2csv.keys())
    classes_all = sorted(data["label"].unique().tolist())
    prob_cols = [f"p:{c}" for c in classes_all]

    for tr_idx, te_idx in skf.split(np.zeros(len(y)), y, groups):
        df_tr = data.iloc[tr_idx]
        df_te = data.iloc[te_idx]
        # Train one classifier per phase on training windows
        clfs = {}
        for ph in phases:
            dtr = df_tr[df_tr["phase"] == ph]
            if dtr.empty:
                continue
            Xtr = dtr[[c for c in dtr.columns if col_is_feature(c)]].apply(pd.to_numeric, errors="coerce")
            ytr = dtr["label"].astype(str)
            if len(pd.unique(ytr)) < 2:
                continue
            clf = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(with_mean=True),
                SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True),
            )
            clf.fit(Xtr, ytr)
            clfs[ph] = clf
        # Aggregate probabilities across windows and phases per trial (test set)
        if not clfs:
            continue
        phases_avail = list(clfs.keys())
        dte = df_te[df_te["phase"].isin(phases_avail)].copy()
        # Predict per window
        prob_cols = [f"p:{c}" for c in classes_all]
        # defer adding prob columns until after predictions to avoid feature-mismatch
        # Collect predictions per phase then merge
        pred_blocks = []
        for ph, clf in clfs.items():
            sub = dte[dte["phase"] == ph]
            if sub.empty:
                continue
            feat_cols = [c for c in sub.columns if col_is_feature(c)]
            Xte = sub[feat_cols].apply(pd.to_numeric, errors="coerce")
            P_small = clf.predict_proba(Xte)
            # map clf.classes_ -> classes_all, fill missing with 0
            cls_small = clf.classes_.tolist()
            map_idx = {c: i for i, c in enumerate(cls_small)}
            P_full = np.zeros((P_small.shape[0], len(classes_all)), dtype=float)
            for j, c in enumerate(classes_all):
                if c in map_idx:
                    P_full[:, j] = P_small[:, map_idx[c]]
                else:
                    P_full[:, j] = 0.0
            blk = pd.DataFrame(P_full, index=sub.index, columns=prob_cols)
            pred_blocks.append(blk)
        if pred_blocks:
            P_all = pd.concat(pred_blocks, axis=0).sort_index()
            dte = dte.join(P_all)
        # trial-wise mean over all windows and phases
        g = dte.groupby("trial_id")
        meanp = g[prob_cols].mean()
        true = g["label"].first()
        idx = np.argmax(meanp.to_numpy(), axis=1)
        pred = [classes_all[i] for i in idx]
        t_bacc.append(balanced_accuracy_score(true, pred))
        t_f1.append(f1_score(true, pred, average="macro"))

    return {
        "trial_bacc_mean": float(np.mean(t_bacc) if t_bacc else np.nan),
        "trial_f1_mean": float(np.mean(t_f1) if t_f1 else np.nan),
    }


def main():
    ap = argparse.ArgumentParser(description="Multi-phase trial aggregation CV")
    ap.add_argument("--gait_full", required=True)
    ap.add_argument("--pre", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--uturn", required=True)
    args = ap.parse_args()
    res = run_cv({
        "gait_full": args.gait_full,
        "pre_uturn": args.pre,
        "post_uturn": args.post,
        "uturn": args.uturn,
    })
    print(res)


if __name__ == "__main__":
    main()
