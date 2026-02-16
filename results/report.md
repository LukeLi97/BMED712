# BMED712 Track A Project 1 — Brief Report

## Overview
This report follows the course objective ‘Robust Gait Phenotyping Across Pathologies (IMU)’. We start with reproducible EDA and visualizations, then expand to robustness benchmarks and explainability.

## Data and Structure
- Data root: `dataset/data`
- Top-level groups: healthy, ortho, neuro; total trials: 1356

## Completed Steps and Figures
- Step 01: Gait events/phases and U-turn visualizations (subset of trials).
- Step 02: Simple features (steps, RMS) and distributions.
- Figures and captions index: `results/figures/figures_index.md`.

## Next Steps (aligned to course goals)
- Expand across cohorts (Healthy/Neuro/Ortho); run 3-class classification and leave-one-group-out benchmarks (subject/pathology/condition).
- Produce sensor ablation curves (#IMUs vs balanced accuracy) and explainability (attention/IG), plus an error modes section.

## Window-based Results (executed 2026-02-16)
We ran 8 configurations: window lengths 3, 4, 5, 6 s with 25% and 50% overlap across 4 phases (pre‑u‑turn, u‑turn, post‑u‑turn, full gait). Subject‑wise 5‑fold CV; groups=subjects; no leakage. Outputs:
- Summary: `results/window_experiments_summary.csv`
- Best RF per phase (balanced accuracy, 25% overlap): `results/window_best_per_phase.csv`
- Per‑phase features: `results/windows/<phase>/features_win{ms}_ov{pct}.csv`

Headline findings (RF sensor):
- pre‑u‑turn: best at 4 s, BAcc≈0.900
- post‑u‑turn: best at 4–6 s, BAcc≈0.890
- gait‑full: best at 5 s, BAcc≈0.868
- u‑turn: challenging; BAcc≈0.60 across tested windows
## Sensor minimization (exhaustive)
- Criterion: minimal k with Macro-F1 within 0.02 of full-sensor Macro-F1 (subject-wise 5-fold).
- Recommended: k=2, sensors=['LB', 'RF'], model=svm (Macro-F1=0.813, BAcc=0.811).

## Condition shift (hardware)
- train on: technoconcept, test on: xsens, LR balanced acc=0.331

## Leave-one-pathology-subtype-out (3-class retrospective check)
- ortho holdout=ACL: recall=0.167 on 60 trials
- ortho holdout=HOA: recall=0.757 on 74 trials
- ortho holdout=KOA: recall=0.679 on 78 trials
- neuro holdout=CIPN: recall=0.898 on 98 trials
- neuro holdout=CVA: recall=0.883 on 128 trials
- neuro holdout=PD: recall=0.825 on 160 trials
- neuro holdout=RIL: recall=0.734 on 398 trials




















<!-- AUTO_REPORT:BEGIN -->
## Executive Summary
- Best full-sensor model: SVM (BAcc≈0.809).
- Recommended minimal sensors: k=2, {LB,RF}, model=SVM (Macro-F1=0.813, BAcc=0.811).

## Step 1 — Baseline (Markdown)
| sensors | model | Macro-F1 (mean±std) | BAcc (mean±std) |
| --- | --- | --- | --- |
| all | LR | 0.748±0.051 | 0.753±0.055 |
| all | RF | 0.810±0.030 | 0.801±0.032 |
| all | SVM | 0.816±0.049 | 0.809±0.052 |
| all | XGB | 0.805±0.028 | 0.792±0.028 |
| feet | LR | 0.728±0.077 | 0.732±0.081 |
| feet | RF | 0.768±0.052 | 0.764±0.056 |
| feet | SVM | 0.793±0.053 | 0.792±0.060 |
| feet | XGB | 0.798±0.030 | 0.797±0.037 |
| he | LR | 0.722±0.051 | 0.732±0.058 |
| he | RF | 0.785±0.038 | 0.774±0.040 |
| he | SVM | 0.786±0.044 | 0.782±0.048 |
| he | XGB | 0.763±0.039 | 0.759±0.041 |
| lb | LR | 0.737±0.047 | 0.747±0.038 |
| lb | RF | 0.709±0.012 | 0.702±0.009 |
| lb | SVM | 0.758±0.043 | 0.763±0.037 |
| lb | XGB | 0.740±0.024 | 0.739±0.032 |
| lf | LR | 0.750±0.056 | 0.765±0.065 |
| lf | RF | 0.753±0.049 | 0.750±0.052 |
| lf | SVM | 0.761±0.057 | 0.769±0.061 |
| lf | XGB | 0.734±0.065 | 0.734±0.068 |
| rf | LR | 0.766±0.064 | 0.771±0.074 |
| rf | RF | 0.761±0.049 | 0.754±0.046 |
| rf | SVM | 0.793±0.041 | 0.792±0.044 |
| rf | XGB | 0.784±0.034 | 0.783±0.037 |


## Step 2 — Sensor Minimization (Markdown)
| k | sensors | Macro-F1 | BAcc |
| --- | --- | --- | --- |
| full | HE,LB,LF,RF | 0.816 | 0.809 |
| 1.000 | RF | 0.793 | 0.792 |
| 2.000 | LB,RF | 0.813 | 0.811 |
| 3.000 | HE,LB,RF | 0.818 | 0.810 |
| recommended(2) | LB,RF | 0.813 | 0.811 |


## Figures
- results/figures/step03_confusion_3class_all.png
- results/figures/step04_sensors_frontier.png
- results/figures/step04b_subset_curve_svm.png

## Repro Commands
- python analysis/train_baseline.py
- python analysis/compile_reports.py
<!-- AUTO_REPORT:END -->




















## Sensor minimization (exhaustive)
- Criterion: minimal k with Macro-F1 within 0.02 of full-sensor Macro-F1 (subject-wise 5-fold).
- Recommended: k=2, sensors=['LB', 'RF'], model=svm (Macro-F1=0.813, BAcc=0.811).

## Condition shift (hardware)
- train on: technoconcept, test on: xsens, LR balanced acc=0.331

## Leave-one-pathology-subtype-out (3-class retrospective check)
- ortho holdout=ACL: recall=0.167 on 60 trials
- ortho holdout=HOA: recall=0.757 on 74 trials
- ortho holdout=KOA: recall=0.679 on 78 trials
- neuro holdout=CIPN: recall=0.898 on 98 trials
- neuro holdout=CVA: recall=0.883 on 128 trials
- neuro holdout=PD: recall=0.825 on 160 trials
- neuro holdout=RIL: recall=0.734 on 398 trials
