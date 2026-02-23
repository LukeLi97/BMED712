# Experiments Log (running notes)

All times in local time, 2026-02-23.

## 12:10 – Frequency windowing (trial 1)
- What: Added fine-grained band-powers as a simple “frequency windowing” (10 bands: 0–2,2–4,…,18–20 Hz) into the existing per-window features pipeline (window_experiments_freq.py).
- Why: Capture cadence and its harmonics with higher resolution than a few wide bands; expected to help non-stationary phases and variable cadence in patient cohorts.
- Data/Setup: RF sensor; phases: gait_full, uturn; windows: 3.0 s (50%), 6.0 s (50%); subject-wise 5-fold CV.
- Result (snapshot):
  - uturn 6.0 s + bands (50%): Balanced Accuracy ≈ 0.932 (LR), consistent with our prior frequency-augmented best.
  - gait_full 3.0 s + bands: run queued (features generation; large CSV). Early runs at 5–6 s show neutral to mild changes vs time-only.
- Next: Finish 3.0 s feature export; add filterbank-style features (triangular weights) if band-powers saturate.

## 12:20 – Mamba baseline (MiniMamba) smoke test
- What: Implemented a lightweight Mamba-style block (CPU/MPS friendly) and quick CV script (analysis/deep/mamba_model.py, mamba_train.py).
- Why: Sequence models may capture non-stationary patterns beyond hand-crafted features, especially in u-turn.
- Data/Setup: For a fast sanity check, used existing window-level feature rows as a 1-step sequence (proxy); RF, gait_full, 3.0 s @ 50%; 5-fold subject-wise CV on MacBook (MPS available).
- Result (sanity): mean acc ≈ 0.43 (very preliminary; expectedly low because inputs are aggregated features, not raw time series).
- Takeaway: Environment/tooling confirmed; to realize gains we must feed raw time-series windows [B, T, C].
- Next: Add a raw-window loader and run MiniMamba on RF time series (start with gait_full 3.0 s, then u-turn 6.0 s).

## 12:30 – Documentation
- Updated brief report with leakage handling (EN) and small-window meaning for patient cohorts (CN/EN).
- Git: commits 025db35 and earlier (docs updates) pushed to origin/main.

