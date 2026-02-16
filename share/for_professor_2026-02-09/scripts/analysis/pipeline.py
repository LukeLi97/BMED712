import os
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

# Ensure repository root is on sys.path so that `dataset` is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Non-interactive backend for saving figures
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dataset.quick_start.load_data import load_trial


TOP_LEVELS = ["healthy", "ortho", "neuro"]


def ensure_dirs(out_dir: Path):
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)


from typing import Optional, List


def find_trials(base_path: Path, limit: Optional[int] = None) -> List[str]:
    trials: List[str] = []
    for top in TOP_LEVELS:
        top_path = base_path / top
        if not top_path.exists():
            continue
        for cohort in sorted(os.listdir(top_path)):
            cohort_path = top_path / cohort
            if not cohort_path.is_dir():
                continue
            for patient in sorted(os.listdir(cohort_path)):
                patient_path = cohort_path / patient
                if not patient_path.is_dir():
                    continue
                for trial in sorted(os.listdir(patient_path)):
                    trial_path = patient_path / trial
                    if trial_path.is_dir():
                        trials.append(trial)
                        if limit is not None and len(trials) >= limit:
                            return trials
    return trials


def save_caption(index_file: Path, rel_path: str, caption: str):
    with index_file.open("a", encoding="utf-8") as f:
        f.write(f"- {rel_path} — {caption}\n")


def step0_inventory(base_path: Path, out_dir: Path):
    inv = {"scanned_at": datetime.now().isoformat(), "structure": {}}
    total_trials = 0
    for top in TOP_LEVELS:
        top_path = base_path / top
        if not top_path.exists():
            continue
        inv["structure"][top] = {}
        for cohort in sorted(os.listdir(top_path)):
            cohort_path = top_path / cohort
            if not cohort_path.is_dir():
                continue
            patients = {}
            for patient in sorted(os.listdir(cohort_path)):
                patient_path = cohort_path / patient
                if not patient_path.is_dir():
                    continue
                trials = [t for t in sorted(os.listdir(patient_path)) if (patient_path / t).is_dir()]
                total_trials += len(trials)
                patients[patient] = trials
            inv["structure"][top][cohort] = patients
    inv["total_trials"] = total_trials
    (out_dir / "artifacts" / "inventory.json").write_text(json.dumps(inv, indent=2), encoding="utf-8")
    return inv


def plot_gait_events(trial_dict: dict, title_prefix: str, save_path: Path):
    md = trial_dict["metadata"]
    data = trial_dict["data_processed"]
    freq = md.get("freq", 100)
    t = data["PacketCounter"].to_numpy(dtype=float) / float(freq)

    fig, ax = plt.subplots(3, figsize=(18, 8), sharex=True, sharey=False, gridspec_kw={'height_ratios': [10, 1, 10]})

    # Top: LF
    if "LF_Gyr_Y" in data.columns:
        ax[0].plot(t, data["LF_Gyr_Y"], label="LF_Gyr_Y")
        ax[0].set_title(f"{title_prefix} — Left Foot (LF)")
        lf_events = md.get("leftGaitEvents")
        if lf_events is not None:
            mi, ma = np.nanmin(data["LF_Gyr_Y"]), np.nanmax(data["LF_Gyr_Y"])
            for to, hs in lf_events:
                ax[0].vlines(to / freq, mi, ma, colors="k", linestyles="--", linewidth=0.8)
                ax[0].vlines(hs / freq, mi, ma, colors="k", linestyles=":", linewidth=0.8)
    ax[0].grid(True, alpha=0.3)

    # Middle: phases
    seg = {
        "gait start": min(np.min(md.get("leftGaitEvents", [[0, 0]])), np.min(md.get("rightGaitEvents", [[0, 0]]))) if md.get("leftGaitEvents") and md.get("rightGaitEvents") else 0,
        "uturn start": md.get("uturnBoundaries", [0, len(data)])[0],
        "uturn end": md.get("uturnBoundaries", [0, len(data)])[1] if len(md.get("uturnBoundaries", [])) > 1 else len(data),
        "gait end": max(np.max(md.get("leftGaitEvents", [[0, 0]])), np.max(md.get("rightGaitEvents", [[len(data)-1, len(data)-1]]))) if md.get("leftGaitEvents") and md.get("rightGaitEvents") else len(data)-1,
    }
    # waiting
    ax[1].add_patch(mpatches.Rectangle((0, 0), seg['gait start'] / freq, 1, alpha=0.1, color="k"))
    ax[1].text(seg['gait start'] / (2 * freq + 1e-6), 0.5, 'waiting', fontsize=8, ha='center', va='center')
    # go
    ax[1].add_patch(mpatches.Rectangle((seg['gait start'] / freq, 0), (seg['uturn start'] - seg['gait start']) / freq, 1, alpha=0.2, color="k"))
    ax[1].text((seg['gait start'] + seg['uturn start']) / (2 * freq + 1e-6), 0.5, 'go', fontsize=8, ha='center', va='center')
    # uturn
    ax[1].add_patch(mpatches.Rectangle((seg['uturn start'] / freq, 0), (seg['uturn end'] - seg['uturn start']) / freq, 1, alpha=0.3, color="k"))
    ax[1].text((seg['uturn start'] + seg['uturn end']) / (2 * freq + 1e-6), 0.5, 'uturn', fontsize=8, ha='center', va='center')
    # back
    ax[1].add_patch(mpatches.Rectangle((seg['uturn end'] / freq, 0), (len(data) - seg['uturn end']) / freq, 1, alpha=0.2, color="k"))
    ax[1].text((seg['uturn end'] + len(data)) / (2 * freq + 1e-6), 0.5, 'back', fontsize=8, ha='center', va='center')
    ax[1].set_yticks([])
    ax[1].set_ylabel('Phases')
    ax[1].grid(False)

    # Bottom: RF
    if "RF_Gyr_Y" in data.columns:
        ax[2].plot(t, data["RF_Gyr_Y"], label="RF_Gyr_Y", color="#C44E52")
        ax[2].set_title(f"{title_prefix} — Right Foot (RF)")
        rf_events = md.get("rightGaitEvents")
        if rf_events is not None:
            mi, ma = np.nanmin(data["RF_Gyr_Y"]), np.nanmax(data["RF_Gyr_Y"])
            for to, hs in rf_events:
                ax[2].vlines(to / freq, mi, ma, colors="k", linestyles="--", linewidth=0.8)
                ax[2].vlines(hs / freq, mi, ma, colors="k", linestyles=":", linewidth=0.8)
    ax[2].grid(True, alpha=0.3)
    ax[2].set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_uturn(trial_dict: dict, title_prefix: str, save_path: Path):
    md = trial_dict["metadata"]
    data = trial_dict["data_processed"]
    freq = md.get("freq", 100)
    t = data["PacketCounter"].to_numpy(dtype=float) / float(freq)
    if "LB_Gyr_X" not in data.columns:
        return False
    angle = (np.cumsum(data["LB_Gyr_X"].to_numpy(dtype=float))) / float(freq)
    angle_deg = angle * 360.0 / (2 * np.pi)
    seg = {
        "uturn start": md.get("uturnBoundaries", [0, len(data)])[0],
        "uturn end": md.get("uturnBoundaries", [0, len(data)])[1] if len(md.get("uturnBoundaries", [])) > 1 else len(data)-1,
    }

    fig, ax = plt.subplots(2, figsize=(18, 6), sharex=True, gridspec_kw={'height_ratios': [10, 1]})
    ax[0].plot(t, angle_deg, color="#4C72B0")
    ax[0].set_title(f"{title_prefix} — LB angle (deg)")
    ax[0].set_ylabel("Angle (°)")
    ax[0].grid(True, alpha=0.3)

    mi = float(np.min(angle_deg))
    ma = float(np.max(angle_deg))
    # go
    ax[0].add_patch(mpatches.Rectangle((0, mi), seg['uturn start'] / freq, ma - mi, alpha=0.2, color="k"))
    ax[1].add_patch(mpatches.Rectangle((0, 0), seg['uturn start'] / freq, 1, alpha=0.2, color="k"))
    # uturn
    ax[0].vlines(seg['uturn start'] / freq, mi, ma, 'black', '--', linewidth=2)
    ax[0].add_patch(mpatches.Rectangle((seg['uturn start'] / freq, mi), (seg['uturn end'] - seg['uturn start']) / freq, ma - mi, alpha=0.3, color="k"))
    ax[1].add_patch(mpatches.Rectangle((seg['uturn start'] / freq, 0), (seg['uturn end'] - seg['uturn start']) / freq, 1, alpha=0.3, color="k"))
    # back
    ax[0].vlines(seg['uturn end'] / freq, mi, ma, 'black', '--', linewidth=2)
    ax[0].add_patch(mpatches.Rectangle((seg['uturn end'] / freq, mi), (len(t) - 1 - seg['uturn end']) / freq, ma - mi, alpha=0.2, color="k"))
    ax[1].add_patch(mpatches.Rectangle((seg['uturn end'] / freq, 0), (len(t) - 1 - seg['uturn end']) / freq, 1, alpha=0.2, color="k"))
    ax[1].set_yticks([])
    ax[1].set_ylabel("Phases")
    ax[1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return True


def step1_eda(base_path: Path, out_dir: Path, trials: List[str], max_plots: int = 6):
    figures_index = out_dir / "figures" / "figures_index.md"
    if figures_index.exists():
        figures_index.unlink()
    figures_index.write_text("# Figures Index\n\n", encoding="utf-8")

    plotted = 0
    for trial_name in trials:
        try:
            trial = load_trial(str(base_path), trial_name)
        except Exception:
            continue
        # Gait events figure
        p1 = out_dir / "figures" / f"step01_gait_events_{trial_name}.png"
        plot_gait_events(trial, f"Trial {trial_name}", p1)
        save_caption(figures_index, str(p1.relative_to(out_dir)), "Gait events and phases (LF/RF)")
        plotted += 1
        # U-turn figure (if available)
        p2 = out_dir / "figures" / f"step01b_uturn_{trial_name}.png"
        ok = plot_uturn(trial, f"Trial {trial_name}", p2)
        if ok:
            save_caption(figures_index, str(p2.relative_to(out_dir)), "U-turn segment and LB angle (cumulative integral of LB_Gyr_X)")
        if plotted >= max_plots:
            break


def extract_simple_features(base_path: Path, trials: List[str]) -> pd.DataFrame:
    rows = []
    for trial_name in trials:
        try:
            t = load_trial(str(base_path), trial_name)
        except Exception:
            continue
        md = t["metadata"]
        df = t["data_processed"]
        freq = md.get("freq", 100)
        duration_s = float(df["PacketCounter"].iloc[-1]) / float(freq)
        lf_events = md.get("leftGaitEvents") or []
        rf_events = md.get("rightGaitEvents") or []
        steps = len(lf_events) + len(rf_events)
        row = {
            "trial": trial_name,
            "duration_s": duration_s,
            "steps": steps,
            "has_LF": "LF_Gyr_Y" in df.columns,
            "has_RF": "RF_Gyr_Y" in df.columns,
            "has_LB": "LB_Gyr_X" in df.columns,
        }
        for sensor in ["LF", "RF"]:
            col = f"{sensor}_Gyr_Y"
            if col in df.columns:
                x = df[col].to_numpy(dtype=float)
                # Time-domain
                row[f"{sensor}_gyr_y_mean"] = float(np.nanmean(x))
                row[f"{sensor}_gyr_y_std"] = float(np.nanstd(x))
                row[f"{sensor}_gyr_y_rms"] = float(np.sqrt(np.nanmean(x**2)))
                # Frequency-domain on full event
                m = np.nanmean(x) if np.isfinite(np.nanmean(x)) else 0.0
                x0 = np.nan_to_num(x - m)
                if x0.size > 1:
                    X = np.fft.rfft(x0)
                    P = (X.real ** 2 + X.imag ** 2)
                    freqs = np.fft.rfftfreq(x0.size, d=1.0 / float(freq))
                    p_sum = float(np.sum(P))
                    if P.size > 0 and p_sum > 0:
                        dom_idx = int(np.argmax(P))
                        row[f"{sensor}_gyr_y_dom_freq_hz"] = float(freqs[dom_idx])
                        row[f"{sensor}_gyr_y_spec_centroid_hz"] = float(np.sum(freqs * P) / p_sum)
                        row[f"{sensor}_gyr_y_spec_power"] = float(p_sum)
                    else:
                        row[f"{sensor}_gyr_y_dom_freq_hz"] = np.nan
                        row[f"{sensor}_gyr_y_spec_centroid_hz"] = np.nan
                        row[f"{sensor}_gyr_y_spec_power"] = np.nan
                else:
                    row[f"{sensor}_gyr_y_dom_freq_hz"] = np.nan
                    row[f"{sensor}_gyr_y_spec_centroid_hz"] = np.nan
                    row[f"{sensor}_gyr_y_spec_power"] = np.nan
            else:
                row[f"{sensor}_gyr_y_mean"] = np.nan
                row[f"{sensor}_gyr_y_std"] = np.nan
                row[f"{sensor}_gyr_y_rms"] = np.nan
                row[f"{sensor}_gyr_y_dom_freq_hz"] = np.nan
                row[f"{sensor}_gyr_y_spec_centroid_hz"] = np.nan
                row[f"{sensor}_gyr_y_spec_power"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def step2_features(base_path: Path, out_dir: Path, trials: List[str]):
    df = extract_simple_features(base_path, trials)
    if df.empty:
        return
    # Save features table
    df.to_csv(out_dir / "artifacts" / "features.csv", index=False)

    # Plot: steps per trial (top-N)
    top = df.sort_values("steps", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(top["trial"], top["steps"], color="#55A868")
    ax.set_title("Step 02 — Steps per trial (Top-10)")
    ax.set_ylabel("Steps")
    ax.set_xlabel("Trial")
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    p = out_dir / "figures" / "step02_steps_per_trial.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)

    idx = out_dir / "figures" / "figures_index.md"
    save_caption(idx, str(p.relative_to(out_dir)), "Steps per trial distribution (Top-10)")

    # Plot: RMS distribution per sensor
    fig, ax = plt.subplots(figsize=(10, 4))
    parts = []
    labels = []
    for sensor in ["LF", "RF"]:
        s = df[f"{sensor}_gyr_y_rms"].dropna()
        if not s.empty:
            parts.append(s.values)
            labels.append(sensor)
    if parts:
        ax.boxplot(parts, labels=labels)
        ax.set_title("Step 02 — LF/RF Gyr_Y RMS distribution")
        ax.set_ylabel("RMS (rad/s)")
        fig.tight_layout()
        p2 = out_dir / "figures" / "step02_rms_boxplot.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        save_caption(idx, str(p2.relative_to(out_dir)), "Inter-trial distribution of LF/RF gyro-Y RMS")


def step3_report(base_path: Path, out_dir: Path, inv: dict):
    report = out_dir / "report.md"
    lines: list[str] = []
    lines.append("# BMED712 Track A Project 1 — Brief Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("This report follows the course objective ‘Robust Gait Phenotyping Across Pathologies (IMU)’. We start with reproducible EDA and visualizations, then expand to robustness benchmarks and explainability.")
    lines.append("")
    lines.append("## Data and Structure")
    lines.append(f"- Data root: `{base_path}`")
    lines.append(f"- Top-level groups: {', '.join([k for k in inv.get('structure', {}).keys()])}; total trials: {inv.get('total_trials', 0)}")
    lines.append("")
    lines.append("## Completed Steps and Figures")
    lines.append("- Step 01: Gait events/phases and U-turn visualizations (subset of trials).")
    lines.append("- Step 02: Simple features (steps, RMS) and distributions.")
    lines.append("- Figures and captions index: `results/figures/figures_index.md`.")
    lines.append("")
    lines.append("## Next Steps (aligned to course goals)")
    lines.append("- Expand across cohorts (Healthy/Neuro/Ortho); run 3-class classification and leave-one-group-out benchmarks (subject/pathology/condition).")
    lines.append("- Produce sensor ablation curves (#IMUs vs balanced accuracy) and explainability (attention/IG), plus an error modes section.")
    report.write_text("\n".join(lines), encoding="utf-8")


def run_all(base_path: str = "dataset/data", out_dir: str = "results", limit: Optional[int] = 12):
    base = Path(base_path)
    out = Path(out_dir)
    ensure_dirs(out)

    inv = step0_inventory(base, out)
    trials = find_trials(base, limit=limit)
    if not trials:
        step3_report(base, out, inv)
        return
    step1_eda(base, out, trials, max_plots=min(6, len(trials)))
    step2_features(base, out, trials)
    step3_report(base, out, inv)


if __name__ == "__main__":
    run_all()
