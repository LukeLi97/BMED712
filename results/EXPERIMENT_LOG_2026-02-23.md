# EXPERIMENT LOG — 2026‑02‑23 (CN/EN)

- 12:55 — Initialized branch `codex/mamba-xgb-windowing`; appended bilingual windowing addendum; created RF vs ALL update.
- 13:00 — Planned XGBoost baseline run and report compilation.
- 13:05 — Planned frequency‑windowing refresh (uturn 6 s @ 50% + bands).
- 13:10 — Added TODOs for Tiny Mamba training (gait_full 3 s @ 50%, uturn 6 s @ 50%).

Next: install xgboost; run `analysis/train_baseline.py` and `analysis/compile_reports.py`.

- 13:25 — Installed xgboost; launched train_baseline (subject-wise CV) and window_experiments sweep (small windows).
- 13:28 — Added Tiny Mamba trainer; attempted mamba-ssm install failed on macOS (no nvcc); fallback GRU path enabled.
- 13:30 — Launched Tiny Mamba (fallback) on gait_full 3 s @ 50% (MPS).
- 13:32 — Created branch and pushed to GitHub: codex/mamba-xgb-windowing.
- 13:45 — Pruned jobs: stopped PIDs 64731, 66271; kept 66214 (mamba-like on gait_full 3s@50%) and 65059 (small-window sweep).
- waiting for PID 66214 to finish before queued runs …
- [paused] User requested to pause all training jobs.
- 19:07 — Resumed: gait_full Mamba (PID 80641); queued pre/post/uturn (PID 80643); launched uturn 6s time+freq eval (PID 80644).
- waiting for PID 80641 to finish before queued runs …
- waiting for PID 92418 to finish before queued runs …
- cleanup — stopped non-essential terminals (freq eval / auto report / window sweep), kept training + queue.
- waiting for PID 96750 to finish before queued runs …
- 11:39 — Restarted training: gait_full PID 96750; queued phases PID 96753.
- waiting for PID 96788 to finish before queued runs …
- orchestrator — will start 12-epoch gait_full, queue pre/post/uturn, and auto-push once the current 1-epoch run finishes.
- waiting for PID 8629 to finish before queued runs …
- 13:36 — Relaunched: gait_full PID 16769; queue PID 16772; auto-push PID 16775.
- waiting for PID 39639 to finish before queued runs …
- start queued run: pre_uturn — win=4.0s ov=25% arch=mamba
- waiting for PID 40146 to finish before queued runs …
- waiting for PID 40840 to finish before queued runs …
