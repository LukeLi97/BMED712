# Window Experiments — Plan and Execution (Week 2)

This note records decisions and artifacts for the sliding‑window study.

## Required 5 Items (now fixed)
- Window length: 3 s, 4 s, 5 s, 6 s.
- Rationale (physiology/signal): Each window covers ~2–6 steps for most adults (≈1–2 Hz step rate), balancing stationarity with sufficient gait-cycle context.
- Overlap: 25% and 50%.
- Overlap rationale: 50% improves temporal resolution and sample count without extreme correlation; 25% provides a sparser, more independent sample set for sensitivity analysis.
- Physiological meaning per window: A short bout of steady walking segment (pre/uturn/post) capturing multiple consecutive steps and inter‑limb coordination.

## Guardrails
- Split subjects first, then window (StratifiedGroupKFold) to prevent leakage.
- Per-window labels inherit from phase segmentation via metadata.
- Feature set: time (mean/std/RMS) + frequency (dominant freq, spectral centroid, total power) per channel.

## Executed Configs (2026‑02‑16)
- 8 runs: 4 lengths × 2 overlaps.
- Output CSV: `results/window_experiments_summary.csv` and `results/window_best_per_phase.csv`.
- Per‑phase features saved under `results/windows/<phase>/features_win{ms}_ov{pct}.csv`.

## Quick Findings
- RF sensor competitive across phases; best RF balanced accuracy by phase (0.25 overlap): pre‑u‑turn 4 s (≈0.900), post‑u‑turn 4–6 s (≈0.890), gait‑full 5–6 s (≈0.868–0.852). U‑turn remains difficult (≈0.60).
