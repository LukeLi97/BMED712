
# Time-Window Analysis — Standalone Report (2026-02-16)

## Executive Summary
- Sliding windows (3, 4, 5, 6 s) with 25% and 50% overlap across four phases markedly improve phase-specific classification versus bulk (full-event) features.
- Best RF-only windows by phase (balanced accuracy, subject-wise 5-fold):
  - pre-u-turn: 4 s @ 25% overlap → BAcc≈0.900
  - post-u-turn: 4 s @ 25% overlap → BAcc≈0.890
  - gait full: 3 s @ 50% overlap → BAcc≈0.891
  - u-turn: 3 s (25%/50%) → BAcc≈0.600 (remains challenging)
- Sensor minimization: RF single-sensor is competitive or better than all sensors in these runs (e.g., gait full RF 0.891 vs ALL 0.884).
- Overlap effect: mixed—50% helps gait full; 25% helps pre/post-u-turn; u-turn unchanged.

## Methods (concise)
- Data: IMU trials under `dataset/data/` with metadata-defined phases (pre-u-turn, u-turn, post-u-turn, gait_full).
- Windowing: 3/4/5/6 s windows; overlaps 25% and 50%; per-phase segmentation via metadata.
- Features (per window, per channel): time-domain (mean, std, RMS) + frequency-domain (dominant freq, spectral centroid, total power).
- Modeling & CV: StratifiedGroupKFold (5 folds) grouped by subject; models LR/SVM/RF; report best by balanced accuracy; single-sensor RF and full-sensor (ALL).
- No leakage: split by subject before evaluation; window labels inherit from phase.

## Results by Phase (key numbers)
- pre-u-turn (RF): best 4 s @ 25% → BAcc≈0.900, Macro-F1≈0.872 (LR)
- post-u-turn (RF): best 4 s @ 25% → BAcc≈0.890, Macro-F1≈0.794 (LR)
- gait full (RF): best 3 s @ 50% → BAcc≈0.891, Macro-F1≈0.809 (LR)
- u-turn (RF/ALL): best 3 s → BAcc≈0.600; needs turn-specific features or longer context

## Comparison to Bulk Baseline
- Baseline (full-event, ALL sensors): BAcc≈0.809.
- Windowed gains (absolute):
  - pre-u-turn: +0.099
  - post-u-turn: +0.081
  - gait full: +0.082
  - u-turn: −0.18 vs baseline subset; windowing here not yet beneficial.

## Interpretation & Takeaways
- Why windows help: capture short, quasi-stationary gait snippets aligning with 2–6 steps, preserving cadence/periodicity and transient asymmetries that bulk averages out.
- Physiological meaning: one window ≈ multiple consecutive steps within a phase, reflecting inter‑limb coordination and local stability.
- Sensor choice: RF alone is strong across phases → promising for IMU minimization.
- Overlap: use 50% when you need finer temporal resolution (gait full); 25% when you want sparser, more independent samples (pre/post turn).

## Limitations & Next Steps
- U-turn remains difficult; try longer windows (8–10 s), curvature/heading-change features, and turn-peak angular velocity metrics.
- Add trial-level aggregation (majority/soft vote over windows) to compare against bulk at the trial granularity.
- Explore step-synchronous (gait-cycle aligned) windows and narrow frequency bands around step/cadence (0.5–3 Hz).

## Reproducibility
- Summary CSV: `results/window_experiments_summary.csv`
- Best per phase (RF): `results/window_best_per_phase.csv`
- Per-phase window features: `results/windows/<phase>/features_win{ms}_ov{pct}.csv`
- Re-run example:
  - `python analysis/window_experiments.py --windows 3.0,4.0,5.0,6.0 --overlap 0.50 --sensors RF,ALL --data dataset/data --out results`
  - `python analysis/window_experiments.py --windows 3.0,4.0,5.0,6.0 --overlap 0.25 --sensors RF,ALL --data dataset/data --out results`

(Models: LR/SVM/RF; metrics are fold means; see the summary CSV for full details.)
