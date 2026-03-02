"""Gait temporal asymmetry analysis.

Computes step/stride time asymmetry from heel-strike events,
performs group-level statistical comparisons (healthy vs pathological),
and produces boxplots, histograms, and time-series visualizations.

Usage:
    python analysis/asymmetry.py
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
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.pipeline import ensure_dirs, find_trials
from dataset.quick_start.load_data import load_trial

# Physiological bounds (seconds)
MIN_STEP_S = 0.2
MAX_STEP_S = 3.0
MIN_STRIDE_S = 0.4
MAX_STRIDE_S = 5.0


# ── helpers ──────────────────────────────────────────────────────────


def extract_heel_strikes(gait_events: list) -> np.ndarray:
    """Return sorted heel-strike sample indices (pair[1])."""
    if not gait_events:
        return np.array([], dtype=int)
    hs = np.array(
        [p[1] for p in gait_events if isinstance(p, (list, tuple)) and len(p) >= 2],
        dtype=int,
    )
    hs.sort()
    return hs


def filter_uturn(hs: np.ndarray, uturn: list) -> np.ndarray:
    """Remove heel strikes inside U-turn boundaries."""
    if hs.size == 0 or len(uturn) < 2:
        return hs
    return hs[~((hs >= uturn[0]) & (hs <= uturn[1]))]


# ── Step 1: validation ──────────────────────────────────────────────


def validate_trial(
    trial_name: str, md: dict, freq: float
) -> Dict:
    """Check event quality for one trial. Returns a dict summary."""
    left_raw = extract_heel_strikes(md.get("leftGaitEvents", []))
    right_raw = extract_heel_strikes(md.get("rightGaitEvents", []))
    uturn = md.get("uturnBoundaries", [])

    left = filter_uturn(left_raw, uturn)
    right = filter_uturn(right_raw, uturn)

    dup_l = len(left) != len(np.unique(left))
    dup_r = len(right) != len(np.unique(right))
    mono_l = bool(np.all(np.diff(left) > 0)) if left.size > 1 else True
    mono_r = bool(np.all(np.diff(right) > 0)) if right.size > 1 else True

    n_bad = 0
    for hs in (left, right):
        if hs.size > 1:
            dt = np.diff(hs) / freq
            n_bad += int(np.sum((dt < MIN_STRIDE_S) | (dt > MAX_STRIDE_S)))

    ok = (
        not dup_l
        and not dup_r
        and mono_l
        and mono_r
        and left.size >= 3
        and right.size >= 3
        and n_bad == 0
    )
    return {
        "trial": trial_name,
        "n_left_hs": int(left.size),
        "n_right_hs": int(right.size),
        "excluded_uturn_left": int(left_raw.size - left.size),
        "excluded_uturn_right": int(right_raw.size - right.size),
        "duplicates": dup_l or dup_r,
        "non_monotonic": not mono_l or not mono_r,
        "n_physio_invalid": n_bad,
        "valid": ok,
    }


# ── Step 2: temporal metrics ────────────────────────────────────────


def _stride_times(hs: np.ndarray, freq: float) -> np.ndarray:
    """Consecutive same-foot HS intervals (seconds)."""
    if hs.size < 2:
        return np.array([])
    dt = np.diff(hs) / freq
    return dt[(dt >= MIN_STRIDE_S) & (dt <= MAX_STRIDE_S)]


def _step_times(
    left_hs: np.ndarray, right_hs: np.ndarray, freq: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Step times: R→next L (left step) and L→next R (right step)."""
    st_left: list[float] = []
    st_right: list[float] = []

    for r in right_hs:
        cands = left_hs[left_hs > r]
        if cands.size > 0:
            dt = (cands[0] - r) / freq
            if MIN_STEP_S <= dt <= MAX_STEP_S:
                st_left.append(dt)

    for l_hs_val in left_hs:
        cands = right_hs[right_hs > l_hs_val]
        if cands.size > 0:
            dt = (cands[0] - l_hs_val) / freq
            if MIN_STEP_S <= dt <= MAX_STEP_S:
                st_right.append(dt)

    return np.array(st_left), np.array(st_right)


def compute_trial_metrics(
    trial_name: str, md: dict, df: pd.DataFrame
) -> Optional[Dict]:
    """Return per-trial temporal metrics dict, or None if insufficient data."""
    freq = float(md.get("freq", 100.0))
    uturn = md.get("uturnBoundaries", [])
    left = filter_uturn(extract_heel_strikes(md.get("leftGaitEvents", [])), uturn)
    right = filter_uturn(extract_heel_strikes(md.get("rightGaitEvents", [])), uturn)

    if left.size < 2 or right.size < 2:
        return None

    step_l, step_r = _step_times(left, right, freq)
    stride_l = _stride_times(left, freq)
    stride_r = _stride_times(right, freq)

    if step_l.size == 0 or step_r.size == 0:
        return None

    return {
        "trial": trial_name,
        "subject": str(md.get("subject", "")),
        "group": str(md.get("group", "unknown")),
        "pathology_key": str(md.get("pathologyKey", "")),
        "freq": freq,
        "step_l": step_l,
        "step_r": step_r,
        "stride_l": stride_l,
        "stride_r": stride_r,
        "mean_step_l": float(np.mean(step_l)),
        "mean_step_r": float(np.mean(step_r)),
        "mean_stride_l": float(np.mean(stride_l)) if stride_l.size else float("nan"),
        "mean_stride_r": float(np.mean(stride_r)) if stride_r.size else float("nan"),
        "left_hs": left,
        "right_hs": right,
    }


# ── Step 3: asymmetry indices ───────────────────────────────────────


def _asym(tl: float, tr: float) -> Dict[str, float]:
    denom = (tl + tr) / 2.0
    return {
        "AI": (tl - tr) / denom if denom > 0 else float("nan"),
        "abs_diff": abs(tl - tr),
        "ratio": tl / tr if tr > 0 else float("nan"),
    }


def _cv(arr: np.ndarray) -> float:
    """Coefficient of variation (std / mean). Returns nan if empty."""
    if arr.size < 2:
        return float("nan")
    m = float(np.mean(arr))
    return float(np.std(arr, ddof=1) / m) if m > 0 else float("nan")


def build_trial_df(metrics_list: List[Dict]) -> pd.DataFrame:
    """One row per trial with asymmetry + variability metrics."""
    rows = []
    for m in metrics_list:
        sa = _asym(m["mean_step_l"], m["mean_step_r"])
        ra = _asym(m["mean_stride_l"], m["mean_stride_r"])
        rows.append(
            {
                "trial": m["trial"],
                "subject": m["subject"],
                "group": m["group"],
                "pathology_key": m["pathology_key"],
                "mean_step_l": m["mean_step_l"],
                "mean_step_r": m["mean_step_r"],
                "mean_stride_l": m["mean_stride_l"],
                "mean_stride_r": m["mean_stride_r"],
                # signed asymmetry
                "step_AI": sa["AI"],
                "stride_AI": ra["AI"],
                # unsigned asymmetry magnitude
                "step_absAI": abs(sa["AI"]),
                "stride_absAI": abs(ra["AI"]),
                # absolute difference
                "step_abs_diff": sa["abs_diff"],
                "stride_abs_diff": ra["abs_diff"],
                # ratio
                "step_ratio": sa["ratio"],
                "stride_ratio": ra["ratio"],
                # within-trial variability (CV of step times)
                "step_CV_l": _cv(m["step_l"]),
                "step_CV_r": _cv(m["step_r"]),
                "stride_CV_l": _cv(m["stride_l"]),
                "stride_CV_r": _cv(m["stride_r"]),
                # average step time (proxy for walking speed)
                "mean_step_time": (m["mean_step_l"] + m["mean_step_r"]) / 2.0,
                # counts
                "n_steps_l": m["step_l"].size,
                "n_steps_r": m["step_r"].size,
                "n_strides_l": m["stride_l"].size,
                "n_strides_r": m["stride_r"].size,
            }
        )
    return pd.DataFrame(rows)


def build_subject_df(df_trial: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trial-level metrics to per-subject means."""
    num_cols = df_trial.select_dtypes(include="number").columns.tolist()
    # exclude count columns from mean aggregation
    agg_cols = [c for c in num_cols if not c.startswith("n_")]
    grp = df_trial.groupby("subject", as_index=False)
    df_subj = grp[agg_cols].mean()
    # attach group label (same for all trials of a subject)
    first = df_trial.groupby("subject")[["group", "pathology_key"]].first()
    df_subj = df_subj.merge(first, on="subject")
    # add trial count
    df_subj["n_trials"] = grp.size()["size"].values
    return df_subj


def build_per_step_df(metrics_list: List[Dict]) -> pd.DataFrame:
    """Long-format: one row per individual step/stride."""
    rows = []
    for m in metrics_list:
        base = {"trial": m["trial"], "subject": m["subject"], "group": m["group"]}
        for v in m["step_l"]:
            rows.append({**base, "side": "left", "metric": "step_time", "value_s": float(v)})
        for v in m["step_r"]:
            rows.append({**base, "side": "right", "metric": "step_time", "value_s": float(v)})
        for v in m["stride_l"]:
            rows.append({**base, "side": "left", "metric": "stride_time", "value_s": float(v)})
        for v in m["stride_r"]:
            rows.append({**base, "side": "right", "metric": "stride_time", "value_s": float(v)})
    return pd.DataFrame(rows)


# ── Step 4: statistics ──────────────────────────────────────────────


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled SD (signed)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return 0.0
    n1, n2 = a.size, b.size
    sp2 = ((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / max(n1 + n2 - 2, 1)
    if sp2 <= 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / np.sqrt(sp2))


def run_stats(df: pd.DataFrame, col: str) -> Dict:
    """t-tests + Cohen's d for one metric column."""
    result: Dict = {"metric": col, "group_stats": {}}

    for g in ("healthy", "ortho", "neuro"):
        vals = df.loc[df["group"] == g, col].dropna().values
        result["group_stats"][g] = {
            "mean": float(np.mean(vals)) if vals.size else float("nan"),
            "std": float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan"),
            "median": float(np.median(vals)) if vals.size else float("nan"),
            "n": int(vals.size),
        }

    # binary: healthy vs pathological
    h = df.loc[df["group"] == "healthy", col].dropna().values
    p = df.loc[df["group"].isin(["ortho", "neuro"]), col].dropna().values
    if h.size >= 2 and p.size >= 2:
        t, pv = stats.ttest_ind(h, p, equal_var=False)
        result["binary"] = {
            "n_healthy": int(h.size),
            "n_pathological": int(p.size),
            "t": float(t),
            "p": float(pv),
            "d": _cohens_d(h, p),
        }

    # pairwise 3-group
    pairs = [("healthy", "ortho"), ("healthy", "neuro"), ("ortho", "neuro")]
    pw = {}
    for g1, g2 in pairs:
        v1 = df.loc[df["group"] == g1, col].dropna().values
        v2 = df.loc[df["group"] == g2, col].dropna().values
        if v1.size >= 2 and v2.size >= 2:
            t, pv = stats.ttest_ind(v1, v2, equal_var=False)
            pw[f"{g1}_vs_{g2}"] = {"t": float(t), "p": float(pv), "d": _cohens_d(v1, v2)}
    result["pairwise"] = pw
    return result


# ── Step 5: plots ───────────────────────────────────────────────────


def plot_boxplot(df: pd.DataFrame, col: str, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 3-group
    groups = ["healthy", "ortho", "neuro"]
    data3 = [df.loc[df["group"] == g, col].dropna().values for g in groups]
    bp = axes[0].boxplot(data3, tick_labels=groups, patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#55A868", "#4C72B0", "#C44E52"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    axes[0].set_title(f"{title} — 3-Group")
    axes[0].set_ylabel(col)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color="grey", linewidth=0.5, linestyle="--")

    # binary
    data_bin = [
        df.loc[df["group"] == "healthy", col].dropna().values,
        df.loc[df["group"].isin(["ortho", "neuro"]), col].dropna().values,
    ]
    bp2 = axes[1].boxplot(data_bin, tick_labels=["healthy", "pathological"], patch_artist=True)
    for patch, c in zip(bp2["boxes"], ["#55A868", "#DD8452"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    axes[1].set_title(f"{title} — Binary")
    axes[1].set_ylabel(col)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color="grey", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_histogram(per_step: pd.DataFrame, path: Path) -> None:
    groups = ["healthy", "ortho", "neuro"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, grp in zip(axes, groups):
        sub = per_step[(per_step["group"] == grp) & (per_step["metric"] == "step_time")]
        left = sub.loc[sub["side"] == "left", "value_s"].values
        right = sub.loc[sub["side"] == "right", "value_s"].values
        bins = np.linspace(0.2, 2.0, 40)
        ax.hist(left, bins=bins, alpha=0.5, label="Left", color="#4C72B0")
        ax.hist(right, bins=bins, alpha=0.5, label="Right", color="#C44E52")
        ax.set_title(f"{grp} (L={left.size}, R={right.size})")
        ax.set_xlabel("Step Time (s)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Count")
    fig.suptitle("Step 06 — Step Time Distribution by Side and Group", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries(
    trial_name: str,
    trial_dict: dict,
    left_hs: np.ndarray,
    right_hs: np.ndarray,
    path: Path,
) -> None:
    md = trial_dict["metadata"]
    data = trial_dict["data_processed"]
    freq = float(md.get("freq", 100.0))
    n = len(data)
    t = np.arange(n) / freq

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    if "LF_Gyr_Y" in data.columns:
        axes[0].plot(t, data["LF_Gyr_Y"].values, alpha=0.7)
        for idx in left_hs:
            if 0 <= idx < n:
                axes[0].axvline(idx / freq, color="blue", ls="--", alpha=0.6, lw=0.8)
        axes[0].set_title(f"{trial_name} — Left Foot (HS marked)")
        axes[0].set_ylabel("Gyr Y (deg/s)")
        axes[0].grid(True, alpha=0.3)

    if "RF_Gyr_Y" in data.columns:
        axes[1].plot(t, data["RF_Gyr_Y"].values, alpha=0.7, color="#C44E52")
        for idx in right_hs:
            if 0 <= idx < n:
                axes[1].axvline(idx / freq, color="red", ls="--", alpha=0.6, lw=0.8)
        axes[1].set_title(f"{trial_name} — Right Foot (HS marked)")
        axes[1].set_ylabel("Gyr Y (deg/s)")
        axes[1].set_xlabel("Time (s)")
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── main pipeline ───────────────────────────────────────────────────


def run(
    base_path: str = "dataset/data",
    out_dir: str = "results",
    limit: Optional[int] = None,
) -> None:
    base = Path(base_path)
    out = Path(out_dir)
    ensure_dirs(out)

    trials = find_trials(base, limit=limit)
    if not trials:
        print("No trials found.")
        return

    validations: list[Dict] = []
    metrics_list: list[Dict] = []
    example: Optional[Dict] = None  # for time-series plot

    for name in trials:
        try:
            trial = load_trial(str(base), name)
        except Exception as exc:
            print(f"  skip {name}: {exc}")
            continue

        md = trial["metadata"]
        freq = float(md.get("freq", 100.0))

        vr = validate_trial(name, md, freq)
        validations.append(vr)

        if not vr["valid"]:
            continue

        m = compute_trial_metrics(name, md, trial["data_processed"])
        if m is not None:
            metrics_list.append(m)
            if example is None:
                example = {"trial": trial, "name": name, "left_hs": m["left_hs"], "right_hs": m["right_hs"]}

    # save validation summary
    val_summary = {
        "total": len(validations),
        "valid": sum(1 for v in validations if v["valid"]),
        "invalid": sum(1 for v in validations if not v["valid"]),
        "with_duplicates": sum(1 for v in validations if v["duplicates"]),
        "non_monotonic": sum(1 for v in validations if v["non_monotonic"]),
    }
    art = out / "artifacts"
    fig_dir = out / "figures"
    (art / "asymmetry_validation.json").write_text(json.dumps(val_summary, indent=2))
    print(f"\nValidation: {val_summary['valid']}/{val_summary['total']} trials valid")

    if not metrics_list:
        print("No valid trials for asymmetry analysis.")
        return

    # build dataframes
    df_trial = build_trial_df(metrics_list)
    df_step = build_per_step_df(metrics_list)
    df_subj = build_subject_df(df_trial)

    df_trial.to_csv(art / "asymmetry_per_trial.csv", index=False)
    df_step.to_csv(art / "asymmetry_per_step.csv", index=False)
    df_subj.to_csv(art / "asymmetry_per_subject.csv", index=False)
    print(f"Trials with metrics: {len(df_trial)}")
    print(f"Unique subjects: {len(df_subj)}")

    # statistics — trial level
    metrics_to_test = (
        "step_AI", "stride_AI",
        "step_absAI", "stride_absAI",
        "step_abs_diff", "stride_abs_diff",
        "step_CV_l", "step_CV_r",
        "stride_CV_l", "stride_CV_r",
        "mean_step_time",
    )
    stats_out: Dict = {}
    for col in metrics_to_test:
        stats_out[col] = run_stats(df_trial, col)
    (art / "asymmetry_stats.json").write_text(json.dumps(stats_out, indent=2, default=str))

    # statistics — subject level (avoids pseudoreplication)
    stats_subj: Dict = {}
    for col in metrics_to_test:
        if col in df_subj.columns:
            stats_subj[col] = run_stats(df_subj, col)
    (art / "asymmetry_stats_subject.json").write_text(json.dumps(stats_subj, indent=2, default=str))

    # print key results (subject-level = more conservative)
    print("\n" + "=" * 60)
    print("SUBJECT-LEVEL RESULTS (n = subjects, not trials)")
    print("=" * 60)
    key_cols = ("step_absAI", "stride_absAI", "step_abs_diff",
                "stride_abs_diff", "step_CV_l", "mean_step_time")
    for col in key_cols:
        s = stats_subj.get(col, {})
        gs = s.get("group_stats", {})
        print(f"\n{col}:")
        for g in ("healthy", "ortho", "neuro"):
            info = gs.get(g, {})
            m_val = info.get("mean", float("nan"))
            s_val = info.get("std", float("nan"))
            print(f"  {g:10s}  {m_val:+.4f} +/- {s_val:.4f}  (n={info.get('n', 0)})")
        b = s.get("binary", {})
        if b:
            sig = "***" if b["p"] < 0.001 else "**" if b["p"] < 0.01 else "*" if b["p"] < 0.05 else "ns"
            print(f"  binary t={b['t']:.3f}, p={b['p']:.4f} {sig}, d={b['d']:.3f}")

    # plots
    plot_boxplot(df_trial, "step_AI", "Step 06 — Step Time Asymmetry Index",
                 fig_dir / "step06_asymmetry_boxplot_AI.png")
    plot_boxplot(df_trial, "step_absAI", "Step 06 — Step Time |AI|",
                 fig_dir / "step06_asymmetry_boxplot_absAI.png")
    plot_boxplot(df_trial, "step_abs_diff", "Step 06 — Step Time |Left - Right|",
                 fig_dir / "step06_asymmetry_boxplot_abs_diff.png")
    plot_boxplot(df_trial, "stride_abs_diff", "Step 06 — Stride Time |Left - Right|",
                 fig_dir / "step06_asymmetry_boxplot_stride_abs_diff.png")
    plot_boxplot(df_trial, "step_CV_l", "Step 06 — Left Step Time CV (Variability)",
                 fig_dir / "step06_asymmetry_boxplot_CV.png")
    plot_boxplot(df_trial, "mean_step_time", "Step 06 — Mean Step Time (Speed Proxy)",
                 fig_dir / "step06_asymmetry_boxplot_speed.png")
    plot_histogram(df_step, fig_dir / "step06_asymmetry_histogram.png")

    if example is not None:
        plot_timeseries(
            example["name"],
            example["trial"],
            example["left_hs"],
            example["right_hs"],
            fig_dir / "step06_asymmetry_timeseries_example.png",
        )

    print(f"\nDone. Figures saved to {fig_dir}")


if __name__ == "__main__":
    run()
