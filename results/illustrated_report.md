# BMED712 Track A Project 1 — Illustrated Report

## Overview
This document summarizes our initial progress toward robust gait phenotyping. It embeds key figures produced by the pipeline and baseline models, and links to numeric artifacts for reproducibility.

## Data and Methods
- Data root: `dataset/data` (healthy, neuro, ortho). Trials contain processed signals and metadata (events, u-turn boundaries).
- EDA: step-wise visualizations and simple features (steps per trial, LF/RF RMS).
- Baseline task: 3-class classification (Healthy / Neuro / Ortho) with subject-group 5-fold CV; sensor ablations (ALL, LF+RF, LF, RF, LB, HE).
- Artifacts: metrics JSON in `results/artifacts/metrics_*.json`.

## Exploratory Visualizations
- Gait events and phases (LF/RF)

  ![Gait events and phases (LF/RF)](figures/step01_gait_events_HS_1_1.png)

- U-turn segment and LB angle

  ![U-turn segment and LB angle](figures/step01b_uturn_HS_1_1.png)

- Steps per trial (Top-10)

  ![Steps per trial (Top-10)](figures/step02_steps_per_trial.png)

- LF/RF gyro-Y RMS distribution

  ![LF/RF gyro-Y RMS distribution](figures/step02_rms_boxplot.png)

## Classification Results (3-class)
- Confusion matrices (sensors = LF,RF)

  ![3-class confusion (LR/RF), sensors=LF,RF](figures/step03_confusion_3class_feet.png)

- Confusion matrices (sensors = ALL)

  ![3-class confusion (LR/RF), sensors=ALL](figures/step03_confusion_3class_all.png)

- Sensor availability frontier (#IMUs vs balanced accuracy)

  ![#IMUs vs balanced accuracy (LR/RF)](figures/step04_sensors_frontier.png)

- RF feature importance (Top-20, sensors = LF,RF)

  ![RF feature importance (Top-20), sensors=LF,RF](figures/step05_importance_3class_feet.png)

## Notes and Next Steps
- Metrics summary is recorded in `results/report.md` and `results/artifacts/metrics_*.json`.
- Next: add leave-one-group-out (subject/pathology/condition) suites to the main results tables; extend to 8-class; integrate explainability (IG/attention) into sensor selection.

---
To regenerate figures and artifacts:
- EDA + visuals: `python3 analysis/pipeline.py`
- Baselines + ablations: `python3 analysis/train_baseline.py`
