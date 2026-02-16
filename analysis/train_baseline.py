import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
from sklearn.svm import SVC


# Ensure repository root is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.pipeline import ensure_dirs, find_trials
from dataset.quick_start.load_data import load_trial


TOP_LEVELS = ["healthy", "ortho", "neuro"]


def collect_features(base_path: Path, trials: List[str]) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    rows: List[Dict] = []
    labels: List[str] = []
    subjects: List[str] = []
    trial_ids: List[str] = []
    pkeys: List[str] = []
    for tr in trials:
        try:
            t = load_trial(str(base_path), tr)
        except Exception:
            continue
        md = t["metadata"]
        df = t["data_processed"]
        fs = float(md.get("freq", 100))
        # Features: numeric columns except PacketCounter
        feats: Dict[str, float] = {}
        for col in df.columns:
            if col == "PacketCounter":
                continue
            if not np.issubdtype(df[col].dtype, np.number):
                continue
            x = df[col].to_numpy(dtype=float)
            # Time-domain
            feats[f"{col}__mean"] = float(np.nanmean(x))
            feats[f"{col}__std"] = float(np.nanstd(x))
            feats[f"{col}__rms"] = float(np.sqrt(np.nanmean(x**2)))
            # Frequency-domain (entire event)
            dom_f, spec_c, spec_p = _spectral_features(x, fs)
            feats[f"{col}__dom_freq_hz"] = float(dom_f)
            feats[f"{col}__spec_centroid_hz"] = float(spec_c)
            feats[f"{col}__spec_power"] = float(spec_p)
        feats["duration_s"] = float(df["PacketCounter"].iloc[-1]) / float(md.get("freq", 100))
        rows.append(feats)
        labels.append(str(md.get("group", "unknown")))
        subjects.append(str(md.get("subject", "")))
        trial_ids.append(tr)
        pkeys.append(str(md.get("pathologyKey", "")))
    X = pd.DataFrame(rows).fillna(0.0)
    y = pd.Series(labels, name="group")
    subj = pd.Series(subjects, name="subject")
    tr_ids = pd.Series(trial_ids, name="trial")
    pkey = pd.Series(pkeys, name="pathologyKey")
    return X, y, subj, tr_ids, pkey


def _spectral_features(x: np.ndarray, fs: float) -> Tuple[float, float, float]:
    """Compute dominant frequency, spectral centroid, and total spectral power.

    Operates on the full signal (no windowing). NaNs are ignored; DC is mitigated
    by mean-centering before FFT.
    """
    if x.size == 0 or fs <= 0:
        return 0.0, 0.0, 0.0
    x = np.asarray(x, dtype=float)
    # Replace NaNs with the column mean (or 0 if all NaN)
    m = np.nanmean(x) if np.isfinite(np.nanmean(x)) else 0.0
    x = np.nan_to_num(x - m, copy=False)
    n = int(x.size)
    if n <= 1:
        return 0.0, 0.0, 0.0
    X = np.fft.rfft(x)
    P = (X.real ** 2 + X.imag ** 2)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    # Guard against empty or all-zero spectrum
    if P.size == 0 or not np.isfinite(P).any():
        return 0.0, 0.0, 0.0
    # Dominant frequency
    idx = int(np.argmax(P))
    dom_freq = float(freqs[idx]) if 0 <= idx < freqs.size else 0.0
    # Spectral centroid
    power_sum = float(np.sum(P))
    spec_centroid = float(np.sum(freqs * P) / power_sum) if power_sum > 0 else 0.0
    # Total spectral power
    spec_power = float(power_sum)
    return dom_freq, spec_centroid, spec_power


def select_feature_columns(X: pd.DataFrame, sensors: List[str]) -> List[str]:
    if sensors == ["ALL"]:
        return list(X.columns)
    cols = []
    for c in X.columns:
        if c == "duration_s":
            cols.append(c)
            continue
        # c like 'LF_Gyr_Y__rms' -> sensor_prefix is before first '__'
        sensor_prefix = c.split("__")[0]
        # sensor name is part before first underscore, e.g., 'LF' from 'LF_Gyr_Y'
        sensor = sensor_prefix.split("_")[0]
        if sensor in sensors:
            cols.append(c)
    return cols


def run_cv_experiment(X: pd.DataFrame, y: pd.Series, subj: pd.Series, out_dir: Path, tag: str):
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    y_enc, classes = pd.factorize(y)
    groups = subj.to_numpy()

    clf_lr = make_pipeline(
        StandardScaler(with_mean=True),
        LogisticRegression(max_iter=200, n_jobs=1, multi_class="auto", class_weight="balanced"),
    )
    clf_rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample")
    clf_svm = make_pipeline(StandardScaler(with_mean=True), SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"))
    # XGBoost (optional)
    if _HAS_XGB:
        clf_xgb = XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob" if len(classes) > 2 else "binary:logistic",
            num_class=int(len(classes)) if len(classes) > 2 else None,
            tree_method="hist",
            eval_metric="mlogloss" if len(classes) > 2 else "logloss",
            random_state=42,
            n_jobs=0,
        )
    metrics = {"lr": [], "rf": [], "svm": []}
    if _HAS_XGB:
        metrics["xgb"] = []
    cms_lr = np.zeros((len(classes), len(classes)), dtype=int)
    cms_rf = np.zeros_like(cms_lr)
    cms_svm = np.zeros_like(cms_lr)
    cms_xgb = np.zeros_like(cms_lr) if _HAS_XGB else None

    for train_idx, test_idx in skf.split(X, y_enc, groups):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y_enc[train_idx], y_enc[test_idx]

        # LR
        clf_lr.fit(Xtr, ytr)
        pred_lr = clf_lr.predict(Xte)
        f1_lr = f1_score(yte, pred_lr, average="macro")
        bacc_lr = balanced_accuracy_score(yte, pred_lr)
        metrics["lr"].append({"macro_f1": f1_lr, "balanced_acc": bacc_lr})
        cms_lr += confusion_matrix(yte, pred_lr, labels=list(range(len(classes))))

        # RF
        clf_rf.fit(Xtr, ytr)
        pred_rf = clf_rf.predict(Xte)
        f1_rf = f1_score(yte, pred_rf, average="macro")
        bacc_rf = balanced_accuracy_score(yte, pred_rf)
        metrics["rf"].append({"macro_f1": f1_rf, "balanced_acc": bacc_rf})
        cms_rf += confusion_matrix(yte, pred_rf, labels=list(range(len(classes))))

        # SVM
        clf_svm.fit(Xtr, ytr)
        pred_svm = clf_svm.predict(Xte)
        f1_svm = f1_score(yte, pred_svm, average="macro")
        bacc_svm = balanced_accuracy_score(yte, pred_svm)
        metrics["svm"].append({"macro_f1": f1_svm, "balanced_acc": bacc_svm})
        cms_svm += confusion_matrix(yte, pred_svm, labels=list(range(len(classes))))

        # XGB (optional)
        if _HAS_XGB:
            clf_xgb.fit(Xtr.to_numpy(), ytr)
            pred_xgb = clf_xgb.predict(Xte.to_numpy())
            f1_xgb = f1_score(yte, pred_xgb, average="macro")
            bacc_xgb = balanced_accuracy_score(yte, pred_xgb)
            metrics["xgb"].append({"macro_f1": f1_xgb, "balanced_acc": bacc_xgb})
            cms_xgb += confusion_matrix(yte, pred_xgb, labels=list(range(len(classes))))

    # Save metrics summary
    summary = {
        "classes": list(map(str, classes)),
        "tag": tag,
        "lr": {
            "macro_f1_mean": float(np.mean([m["macro_f1"] for m in metrics["lr"]])),
            "macro_f1_std": float(np.std([m["macro_f1"] for m in metrics["lr"]])),
            "balanced_acc_mean": float(np.mean([m["balanced_acc"] for m in metrics["lr"]])),
            "balanced_acc_std": float(np.std([m["balanced_acc"] for m in metrics["lr"]])),
        },
        "rf": {
            "macro_f1_mean": float(np.mean([m["macro_f1"] for m in metrics["rf"]])),
            "macro_f1_std": float(np.std([m["macro_f1"] for m in metrics["rf"]])),
            "balanced_acc_mean": float(np.mean([m["balanced_acc"] for m in metrics["rf"]])),
            "balanced_acc_std": float(np.std([m["balanced_acc"] for m in metrics["rf"]])),
        },
        "svm": {
            "macro_f1_mean": float(np.mean([m["macro_f1"] for m in metrics["svm"]])),
            "macro_f1_std": float(np.std([m["macro_f1"] for m in metrics["svm"]])),
            "balanced_acc_mean": float(np.mean([m["balanced_acc"] for m in metrics["svm"]])),
            "balanced_acc_std": float(np.std([m["balanced_acc"] for m in metrics["svm"]])),
        },
    }
    if _HAS_XGB:
        summary["xgb"] = {
            "macro_f1_mean": float(np.mean([m["macro_f1"] for m in metrics["xgb"]])),
            "macro_f1_std": float(np.std([m["macro_f1"] for m in metrics["xgb"]])),
            "balanced_acc_mean": float(np.mean([m["balanced_acc"] for m in metrics["xgb"]])),
            "balanced_acc_std": float(np.std([m["balanced_acc"] for m in metrics["xgb"]])),
        }
    (out_dir / "artifacts" / f"metrics_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot confusion matrices
    ncols = 4 if _HAS_XGB else 3
    fig, ax = plt.subplots(1, ncols, figsize=(5*ncols, 5))
    ConfusionMatrixDisplay(cms_lr, display_labels=classes).plot(ax=ax[0], colorbar=False)
    ax[0].set_title(f"LR Confusion — {tag}")
    ConfusionMatrixDisplay(cms_rf, display_labels=classes).plot(ax=ax[1], colorbar=False)
    ax[1].set_title(f"RF Confusion — {tag}")
    ConfusionMatrixDisplay(cms_svm, display_labels=classes).plot(ax=ax[2], colorbar=False)
    ax[2].set_title(f"SVM Confusion — {tag}")
    if _HAS_XGB and cms_xgb is not None:
        ConfusionMatrixDisplay(cms_xgb, display_labels=classes).plot(ax=ax[3], colorbar=False)
        ax[3].set_title(f"XGB Confusion — {tag}")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"step03_confusion_{tag}.png", dpi=150)
    plt.close(fig)

    return summary


def available_sensors_from_features(X: pd.DataFrame) -> List[str]:
    sensors = set()
    for c in X.columns:
        if c == "duration_s":
            continue
        s = c.split("__")[0].split("_")[0]
        if s in {"HE", "LB", "LF", "RF"}:
            sensors.add(s)
    return sorted(sensors)


def run_sensor_subset_exhaustive(
    X: pd.DataFrame,
    y: pd.Series,
    subj: pd.Series,
    out_dir: Path,
    base_tag: str,
    model: str = "rf",
    delta_f1_max: float = 0.02,
):
    """Exhaustive subset search over sensors (4->3->2->1) using the chosen model.

    Picks the minimal-k subset whose mean Macro-F1 is within `delta_f1_max` of the full-sensor mean.
    Saves metrics and a small plot.
    """
    from itertools import combinations

    sensors_all = available_sensors_from_features(X)
    if not sensors_all:
        return None

    # Helper to evaluate a specific sensor set with the selected model
    def eval_sensors(sensors: List[str]):
        cols = select_feature_columns(X, sensors)
        if not cols:
            return None
        # Reuse core CV but only keep the needed model metrics
        summary = run_cv_experiment(X[cols], y, subj, out_dir, f"{base_tag}_{''.join(sensors).lower()}")
        return {
            "macro_f1": summary[model]["macro_f1_mean"],
            "macro_f1_std": summary[model]["macro_f1_std"],
            "bacc": summary[model]["balanced_acc_mean"],
            "bacc_std": summary[model]["balanced_acc_std"],
        }

    # Full sensors baseline
    full_metrics = eval_sensors(["ALL"])  # select_feature_columns understands ALL
    if full_metrics is None:
        return None

    results: Dict[str, Dict] = {"full": {"sensors": sensors_all, **full_metrics}}
    best_per_k: Dict[int, Dict] = {}

    for k in [3, 2, 1]:
        best = None
        for comb in combinations(sensors_all, k):
            m = eval_sensors(list(comb))
            if m is None:
                continue
            entry = {"sensors": list(comb), **m}
            results[f"k{k}_{'_'.join(comb).lower()}"] = entry
            if best is None or m["macro_f1"] > best["macro_f1"]:
                best = entry
        if best:
            best_per_k[k] = best

    # Choose minimal k meeting the delta criterion
    recommended = None
    for k in [1, 2, 3]:
        b = best_per_k.get(k)
        if not b:
            continue
        if b["macro_f1"] >= (full_metrics["macro_f1"] - delta_f1_max):
            recommended = {"k": k, **b}
            break
    if recommended is None:
        # If nothing meets the threshold, pick the best among k=3
        if 3 in best_per_k:
            recommended = {"k": 3, **best_per_k[3]}

    # Save JSON summary
    out = {
        "model": model,
        "delta_f1_max": delta_f1_max,
        "sensors_all": sensors_all,
        "full": full_metrics,
        "best_per_k": best_per_k,
        "recommended": recommended,
    }
    (out_dir / "artifacts" / f"step04b_subset_search_{model}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Plot performance vs k (use best per k)
    if best_per_k:
        ks = sorted(best_per_k.keys())
        f1s = [best_per_k[k]["macro_f1"] for k in ks]
        bas = [best_per_k[k]["bacc"] for k in ks]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ks, f1s, marker='o', label='Macro-F1')
        ax.plot(ks, bas, marker='s', label='Balanced Acc')
        ax.axhline(full_metrics["macro_f1"] - delta_f1_max, color='r', linestyle='--', alpha=0.6, label='F1 threshold')
        ax.set_xlabel('# IMUs (k)')
        ax.set_ylabel('Score')
        ax.set_title(f'Step 04b — Best subset per k ({model})')
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        p = out_dir / "figures" / f"step04b_subset_curve_{model}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
    return out


def plot_importance(X: pd.DataFrame, model: RandomForestClassifier, out: Path, tag: str, topk: int = 20):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return
    idx = np.argsort(importances)[::-1][:topk]
    names = [X.columns[i] for i in idx]
    vals = importances[idx]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(vals)), vals[::-1])
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels([n for n in names[::-1]])
    ax.set_title(f"RF Feature Importance — {tag}")
    fig.tight_layout()
    fig.savefig(out / "figures" / f"step05_importance_{tag}.png", dpi=150)
    plt.close(fig)


def save_caption(index_file: Path, rel_path: str, caption: str):
    with index_file.open("a", encoding="utf-8") as f:
        f.write(f"- {rel_path} — {caption}\n")


def run_all(base_path: str = "dataset/data", out_dir: str = "results"):
    base = Path(base_path)
    out = Path(out_dir)
    ensure_dirs(out)

    trials = find_trials(base, limit=None)
    X, y, subj, tr_ids, pkey = collect_features(base, trials)
    # Sensor ablation settings
    sensor_sets = [
        ("all", ["ALL"]),
        ("feet", ["LF", "RF"]),
        ("lb", ["LB"]),
        ("lf", ["LF"]),
        ("rf", ["RF"]),
        ("he", ["HE"]),
    ]

    idx_md = out / "figures" / "figures_index.md"
    if not idx_md.exists():
        idx_md.parent.mkdir(parents=True, exist_ok=True)
        idx_md.write_text("# Figures Index\n\n", encoding="utf-8")

    # local report path for any notes appended in this script
    report = out / "report.md"
    summaries = []
    for tag, sensors in sensor_sets:
        cols = select_feature_columns(X, sensors)
        if not cols:
            continue
        Xs = X[cols]
        summary = run_cv_experiment(Xs, y, subj, out, f"3class_{tag}")
        summaries.append((tag, summary))

        # Train RF on full data for importance plot
        y_enc, classes = pd.factorize(y)
        rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample")
        rf.fit(Xs, y_enc)
        plot_importance(Xs, rf, out, f"3class_{tag}")

        # Append to figures index
        save_caption(idx_md, str(Path("figures") / f"step03_confusion_3class_{tag}.png"), f"3-class confusion matrices (LR/RF, sensors={','.join(sensors)})")
        save_caption(idx_md, str(Path("figures") / f"step05_importance_3class_{tag}.png"), f"RF feature importance (Top-20, sensors={','.join(sensors)})")

    # Note: Do not append baselines to report here to avoid duplicates.
    # The consolidated report (tables + summary) is handled by analysis/compile_reports.py.

    # Step 4 — sensor frontier plot (accuracy vs #IMUs)
    def num_sensors(name: str) -> int:
        return {"he": 1, "lb": 1, "lf": 1, "rf": 1, "feet": 2, "all": 4}.get(name, 0)

    xs = []
    lr_ba = []
    rf_ba = []
    for tag, summary in summaries:
        xs.append(num_sensors(tag))
        lr_ba.append(summary["lr"]["balanced_acc_mean"])
        rf_ba.append(summary["rf"]["balanced_acc_mean"]) 
    order = np.argsort(xs)
    xs_sorted = np.array(xs)[order]
    lr_sorted = np.array(lr_ba)[order]
    rf_sorted = np.array(rf_ba)[order]
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(xs_sorted, lr_sorted, marker='o', label='LR')
    ax.plot(xs_sorted, rf_sorted, marker='s', label='RF')
    ax.set_xlabel('# IMUs')
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Step 04 — Sensor Availability Frontier (3-class)')
    ax.set_xticks(sorted(set(xs_sorted.tolist())))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fpath = out / "figures" / "step04_sensors_frontier.png"
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    save_caption(idx_md, str(Path("figures") / fpath.name), "#IMUs vs balanced accuracy (LR/RF)")

    # Step 4b — exhaustive subset search using the top model on full sensors
    # Choose top model by best balanced acc on full sensors
    full_summary = next((s for t, s in summaries if t == "all"), None)
    top_model = "rf"
    if full_summary is not None:
        # pick the model with highest balanced_acc_mean among lr/rf/svm(/xgb)
        ba = {
            "lr": full_summary["lr"]["balanced_acc_mean"],
            "rf": full_summary["rf"]["balanced_acc_mean"],
            "svm": full_summary["svm"]["balanced_acc_mean"],
        }
        if _HAS_XGB and "xgb" in full_summary:
            ba["xgb"] = full_summary["xgb"]["balanced_acc_mean"]
        top_model = max(ba.items(), key=lambda kv: kv[1])[0]

    subset_out = run_sensor_subset_exhaustive(X, y, subj, out, base_tag="3class_subsets", model=top_model, delta_f1_max=0.02)
    if subset_out is not None:
        rec = subset_out.get("recommended") or {}
        with report.open("a", encoding="utf-8") as f:
            f.write("\n## Sensor minimization (exhaustive)\n")
            f.write(
                "- Criterion: minimal k with Macro-F1 within 0.02 of full-sensor Macro-F1 (subject-wise 5-fold).\n"
            )
            f.write(
                f"- Recommended: k={rec.get('k')}, sensors={rec.get('sensors')}, model={subset_out.get('model')} "
                f"(Macro-F1={rec.get('macro_f1', float('nan')):.3f}, BAcc={rec.get('bacc', float('nan')):.3f}).\n"
            )

    # Step 5 — condition shift (hardware vendor) if available
    # Build vendor labels
    vendors = []
    # Recover vendor info by reloading minimal metadata per trial
    vendor_map: Dict[str, str] = {}
    for tr in tr_ids:
        try:
            t = load_trial(str(base), tr)
        except Exception:
            continue
        s = str(t["metadata"].get("sensor", "")).lower()
        v = "xsens" if "xsens" in s else ("technoconcept" if "technoconcept" in s else s.strip())
        vendor_map[tr] = v
    vendor_series = tr_ids.map(vendor_map).fillna("")
    uniq_vendors = sorted(set(vendor_series.tolist()))
    if len([v for v in uniq_vendors if v]) >= 2:
        # choose two main vendors
        counts = vendor_series.value_counts()
        major, minor = counts.index[:2].tolist()
        # Use best sensor setting (feet)
        cols = select_feature_columns(X, ["LF","RF"]) if set(["feet"]) else X.columns
        y_enc, classes = pd.factorize(y)
        lr = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=200, n_jobs=1))
        # Train on major vendor only
        train_mask = vendor_series == major
        lr.fit(X[cols][train_mask], y_enc[train_mask.to_numpy()])
        # Test on minor vendor
        test_mask = vendor_series == minor
        pred = lr.predict(X[cols][test_mask])
        bacc = balanced_accuracy_score(y_enc[test_mask.to_numpy()], pred)
        with report.open("a", encoding="utf-8") as f:
            f.write(f"\n## Condition shift (hardware)\n- train on: {major}, test on: {minor}, LR balanced acc={bacc:.3f}\n")

    # Step 6 — leave-one-pathology-subtype-out (report recall on held-out subtype)
    def leave_one_pathology_eval(group_name: str) -> List[str]:
        notes: List[str] = []
        pks = sorted(set(pkey[y == group_name].tolist()))
        for pk in pks:
            holdout_mask = (y == group_name) & (pkey == pk)
            if holdout_mask.sum() < 10:  # too few samples; skip to keep stable
                continue
            train_mask = ~(holdout_mask)
            # use best-performing sensor setting from summaries (feet by default)
            cols = select_feature_columns(X, ["LF","RF"]) if set(["feet"]) else X.columns
            Xtr, Xte = X[cols][train_mask], X[cols]
            y_enc, classes = pd.factorize(y)
            lr = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=200, n_jobs=1))
            lr.fit(Xtr, y_enc[train_mask.to_numpy()])
            pred = lr.predict(X[cols])
            # recall for the held-out group on its held-out subtype
            target_idx = np.where(classes == group_name)[0][0]
            recall = (pred[holdout_mask.to_numpy()] == target_idx).mean()
            notes.append(f"{group_name} holdout={pk}: recall={recall:.3f} on {int(holdout_mask.sum())} trials")
        return notes

    notes_ortho = leave_one_pathology_eval("ortho")
    notes_neuro = leave_one_pathology_eval("neuro")
    if notes_ortho or notes_neuro:
        with report.open("a", encoding="utf-8") as f:
            f.write("\n## Leave-one-pathology-subtype-out (3-class retrospective check)\n")
            for n in notes_ortho + notes_neuro:
                f.write(f"- {n}\n")


if __name__ == "__main__":
    run_all()
