"""Summarize per-phase, per-sensor discriminability using single-sensor features.

Outputs results/sensor_phase_summary.csv with columns:
- phase: pre_uturn | uturn | post_uturn | gait_full
- sensor: HE | LB | LF | RF
- n_trials: number of rows used in the phase CSV
- best_feature: feature name within that sensor achieving highest pairwise-d (avg)
- best_score: average absolute Cohen's d across label pairs for best_feature
- mean_score: mean of average-d across all features of that sensor
- median_score: median of average-d across all features of that sensor
- top3_features: semicolon-separated feature names with highest scores

Notes
- Labels expected: 'Healthy', 'Ortho', 'Neuro' (case sensitive, from export_features.to_label).
- No external deps beyond numpy/pandas.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


LABELS = ("Healthy", "Ortho", "Neuro")
SENSORS = ("HE", "LB", "LF", "RF")
TAIL_COLS = ("trial_id", "subject_id", "label")


def _pairwise_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute Cohen's d with pooled SD; returns 0 if degenerate.

    Inputs are 1D arrays after dropping NaNs.
    """
    a = a.astype(float)
    b = b.astype(float)
    if a.size < 2 or b.size < 2:
        return 0.0
    ma = float(np.mean(a))
    mb = float(np.mean(b))
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    # pooled SD
    n1, n2 = a.size, b.size
    sp2 = ((n1 - 1) * va + (n2 - 1) * vb) / max(n1 + n2 - 2, 1)
    if sp2 <= 0:
        return 0.0
    d = (ma - mb) / np.sqrt(sp2)
    return float(abs(d))


def _avg_pairwise_d(values: np.ndarray, labels: Sequence[str]) -> float:
    """Average pairwise Cohen's d across label pairs for a single feature."""
    scores: List[float] = []
    for i in range(len(LABELS)):
        for j in range(i + 1, len(LABELS)):
            la, lb = LABELS[i], LABELS[j]
            mask = (labels == la) | (labels == lb)
            x = values[mask]
            y = labels[mask]
            a = x[y == la]
            b = x[y == lb]
            a = a[~np.isnan(a)]
            b = b[~np.isnan(b)]
            if a.size and b.size:
                scores.append(_pairwise_cohens_d(a, b))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def _feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c not in TAIL_COLS]
    # Drop entirely-NaN columns
    cols = [c for c in cols if not df[c].isna().all()]
    return cols


def summarize_phase(df: pd.DataFrame, phase: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if df.empty:
        return rows
    all_cols = _feature_columns(df)
    if not all_cols:
        return rows
    labels = df["label"].astype(str).to_numpy()

    for sensor in SENSORS:
        sensor_cols = [c for c in all_cols if c.startswith(sensor + "_")]
        if not sensor_cols:
            continue
        scores: List[Tuple[str, float]] = []
        for col in sensor_cols:
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy()
            s = _avg_pairwise_d(vals, labels)
            scores.append((col, s))
        if not scores:
            continue
        scores.sort(key=lambda t: t[1], reverse=True)
        best_feature, best_score = scores[0]
        mean_score = float(np.mean([s for _, s in scores]))
        median_score = float(np.median([s for _, s in scores]))
        top3 = ";".join([name for name, _ in scores[:3]])
        rows.append(
            {
                "phase": phase,
                "sensor": sensor,
                "n_trials": int(len(df)),
                "best_feature": best_feature,
                "best_score": best_score,
                "mean_score": mean_score,
                "median_score": median_score,
                "top3_features": top3,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize single-sensor performance per phase")
    ap.add_argument("--pre", default=os.path.join("results", "features_pre_uturn.csv"))
    ap.add_argument("--uturn", default=os.path.join("results", "features_uturn.csv"))
    ap.add_argument("--post", default=os.path.join("results", "features_post_uturn.csv"))
    ap.add_argument("--gait", default=os.path.join("results", "features_gait_full.csv"))
    ap.add_argument("--out", default=os.path.join("results", "sensor_phase_summary.csv"))
    args = ap.parse_args()

    rows: List[Dict[str, object]] = []
    for phase, path in (
        ("pre_uturn", args.pre),
        ("uturn", args.uturn),
        ("post_uturn", args.post),
        ("gait_full", args.gait),
    ):
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        rows.extend(summarize_phase(df, phase))

    if not rows:
        raise SystemExit("No input CSVs found; run export_features first.")

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    # Also emit a compact best-per-phase summary (CSV-like txt)
    best_lines: List[str] = []
    for phase in ("pre_uturn", "uturn", "post_uturn", "gait_full"):
        sub = out_df[out_df["phase"] == phase]
        if sub.empty:
            continue
        best = sub.sort_values("best_score", ascending=False).iloc[0]
        best_lines.append(
            f"{phase}: sensor={best['sensor']}, best_score={best['best_score']:.3f}, feature={best['best_feature']}"
        )
    with open(os.path.join(os.path.dirname(args.out), "sensor_phase_best.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(best_lines))


if __name__ == "__main__":
    main()

