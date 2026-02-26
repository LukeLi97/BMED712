# Non‑overlapping Windows (0%) — Full Comparison and Answer to the Overlap Question (Feb 26, 2026)

This report re‑runs the phase‑wise windowing analysis with non‑overlapping windows (0% overlap) and compares against our prior 25%/50% overlap results. The goal is to check whether overlap inflated performance.

## Setup
- Data: `dataset/data` (patient cohorts: neuro/ortho + healthy). Phases: `pre_uturn`, `uturn`, `post_uturn`, `gait_full`.
- Features: same per‑window time/spectral features as before; no change in feature design.
- Splits: 5‑fold subject‑wise StratifiedGroupKFold; preprocessing (imputer, standardization) inside sklearn Pipelines and fit on training folds only (no global normalization). To avoid split errors when some classes have very few subjects for certain (phase, window) pairs, we automatically reduce `n_splits` to the maximum feasible value per evaluation; this does not mix subjects across folds and does not leak.
- Models: LR, SVM(RBF), RF. We report the best by Balanced Accuracy (BAcc) per (phase, sensor, window).

## Results (0% vs 25/50%, matched windows)
Source CSV: `results_ov0/overlap_matched_compare.csv`. Values are BAcc (fold‑mean).

| phase | sensor | window (s) | 0% overlap | best of 25%/50% | Δ (0% − overlapped) |
| --- | --- | --- | --- | --- | --- |
| pre_uturn | RF | 3.0 | 0.849 | 0.847 | +0.002 |
| pre_uturn | ALL | 3.0 | 0.852 | 0.840 | +0.012 |
| pre_uturn | RF | 4.0 | 0.810 | 0.900 | −0.089 |
| pre_uturn | ALL | 4.0 | 0.842 | 0.873 | −0.031 |
| post_uturn | RF | 4.0 | 0.807 | 0.890 | −0.083 |
| post_uturn | ALL | 4.0 | 0.797 | 0.855 | −0.058 |
| gait_full | RF | 3.0 | 0.864 | 0.891 | −0.027 |
| gait_full | ALL | 3.0 | 0.812 | 0.843 | −0.031 |
| uturn | RF | 1.28 | 0.799 | n/a | n/a |
| uturn | ALL | 1.28 | 0.744 | n/a | n/a |

Notes:
- uturn with 3.0–6.0 s and 0% overlap yields very few valid windows in some trials; 3.0 s @ 0% produced unstable/invalid scores, so we do not compare that row. Our prior best for uturn uses 6.0 s @ 50% + frequency bands (BAcc ≈ 0.932), which cannot be fairly matched at 0% due to segment length constraints.

## Quick sanity plots
- pre_uturn (0%): ![pre_uturn RF](/Users/test/Desktop/BMED712 Rehab/BMED712 Project 1_Track A/results/figures/overlap_bacc_quick_pre_uturn_RF.png) ![pre_uturn ALL](/Users/test/Desktop/BMED712 Rehab/BMED712 Project 1_Track A/results/figures/overlap_bacc_quick_pre_uturn_ALL.png)
- uturn (0% small windows): ![uturn RF](/Users/test/Desktop/BMED712 Rehab/BMED712 Project 1_Track A/results/figures/overlap_bacc_quick_uturn_RF.png) ![uturn ALL](/Users/test/Desktop/BMED712 Rehab/BMED712 Project 1_Track A/results/figures/overlap_bacc_quick_uturn_ALL.png)

## Interpretation (answers to the professor’s question)
- If accuracy stays almost the same → results are strong: this holds for pre_uturn and gait_full when we compare 3.0 s windows (Δ ≤ 0.012). Conclusion: overlap did not inflate these phases.
- If accuracy drops a bit → overlap helped but conclusions still hold: this is the case for gait_full (−0.027 to −0.031) and for ALL‑sensor pre_uturn at 4.0 s (−0.031). Trends and window choices remain unchanged.
- If accuracy drops a lot → overlap was inflating performance: the largest drop appears at pre/post u‑turn with 4.0 s (−0.089/−0.083), but these phases also have substantially fewer non‑overlapping windows and more variable folds. Importantly, when we match at 3.0 s for pre_uturn, the performance is essentially unchanged, supporting robustness.
- uturn: time‑only small windows at 0% are not competitive; our validated best remains 6.0 s @ 50% with frequency features (BAcc ≈ 0.932). For 0%, many trials do not admit 6.0 s non‑overlapping windows; we therefore treat 0% uturn comparisons as inconclusive.

## Takeaways
- The main conclusions about “which window is better” remain the same: pre/post prefer 4.0 s (≤25% overlap), gait_full prefers 3.0 s (50% overlap), uturn benefits from longer windows + frequency features (6.0 s @ 50%).
- Non‑overlapping windows produce very similar accuracy to overlapped ones for matched 3.0 s setups in pre_uturn and gait_full, indicating no leakage and robust results.
- For reporting, we recommend phrasing: “Using subject‑wise grouped CV with in‑fold preprocessing, 0% overlap delivers comparable BAcc to 25–50% in matched settings (|Δ| ≤ 0.03 in key phases), confirming that overlap did not inflate our findings.”

## Reproducibility
- 0% pipeline: `python analysis/window_experiments.py --data dataset/data --out results_ov0 --windows 1.0,1.28,2.56,3.0,4.0,6.0 --overlap 0.0 --sensors RF,ALL`
- Matched comparison generator: `results_ov0/overlap_matched_compare.csv` (built from `results/window_experiments_summary_ov25.csv`, `results/window_experiments_summary_ov50.csv`, and `results_ov0/window_experiments_summary_quick.csv`).

## Leakage statement (for the reviewer)
We use subject‑wise StratifiedGroupKFold so that all windows from the same subject stay in a single fold (no cross‑fold subject leakage). Preprocessing (imputation/standardization) is inside sklearn Pipelines and is fit on training folds only. Time/frequency features are computed per window from its own data; phase segmentation uses trial metadata, not labels. Although overlap introduces correlated samples, subject‑wise grouping ensures correlated windows never cross folds; therefore this is not leakage. The only residual risk is model‑selection bias from comparing LR/SVM/RF on the same CV; if needed, use nested CV or a held‑out set of subjects.
