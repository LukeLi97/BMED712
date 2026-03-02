# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Clinical gait analysis research project (BMED712 Track A). Classifies gait pathology from IMU sensor data across three cohorts: healthy, neurological, and orthopedic. The project benchmarks ML model robustness across pathologies, sensor availability, and acquisition conditions.

## Common Commands

```bash
# Activate virtual environment (Python 3.9)
cd "BMED712 Project 1_Track A"
source .venv/bin/activate

# Install dependencies
pip install numpy pandas matplotlib scikit-learn scipy xgboost torch

# Quick sanity check — load a trial
python -c "from dataset.quick_start.load_data import load_trial; t=load_trial('dataset/data','KOA_2_1'); print(t['metadata'].keys())"

# Run tests
pytest -q

# Export features (segments: gait, full, pre_uturn, uturn, post_uturn)
python -m dataset.quick_start.export_features \
  --data-root dataset/data --segment gait \
  --template-csv master_features.csv --out results/features_gait.csv

# Run baseline models (LR, RF, SVM, XGB)
python analysis/train_baseline.py

# Run deep learning (TinyMambaTS)
python analysis/train_mamba_windows.py \
  --data dataset/data --phase gait_full \
  --win 3.0 --overlap 0.5 --device mps --out results/artifacts

# Compile reports
python analysis/compile_reports.py
```

## Architecture

```
dataset/
  data/<cohort>/<subject>/<trial>/   Raw data (do NOT rename files)
  quick_start/                       Data loading, feature extraction, plotting
    load_data.py                     load_trial(), load_bdd() — entry points
    features.py                      compute_trial_features(), per-channel stats + FFT
    export_features.py               CLI batch export to CSV
    plot_data.py                     Gait event visualization
    summarize_sensors.py             Sensor summary statistics

analysis/
  train_baseline.py                  Traditional ML: LR, RF, SVM, XGB with StratifiedGroupKFold
  train_mamba_windows.py             Deep learning: TinyMambaTS (Mamba/GRU) on raw windows
  pipeline.py                        Shared utilities: find_trials(), plot_gait_events()
  windowing.py                       Window extraction: get_phase_bounds(), iter_windows()
  frequency_features.py              Bandpower, band ratios, spectral features
  compile_reports.py                 Aggregate results into markdown/HTML reports
  models/mamba_ts.py                 TinyMambaTS model (Conv1d stem + Mamba/GRU + dense head)
  deep/                              Lightweight Mamba surrogate, raw window datasets

configs/                             YAML experiment configs (e.g., week1_baseline.yaml)
results/                             Output artifacts, figures, CSVs, reports
```

## Key Conventions

- **4 sensors**: HE (Head), LB (Lower Back), LF (Left Foot), RF (Right Foot); each has FreeAcc and Gyr on X/Y/Z axes
- **Trial naming**: `<COHORT_ID>_<SUBJECT_ID>_<TRIAL_ID>` (e.g., `KOA_2_1`)
- **Per-trial files**: `*_raw_data_<SENSOR>.txt` (tab-delimited), `*_processed_data.txt`, `*_meta.json`
- **Do not alter** dataset filename patterns or directory depth — loaders depend on these
- **Cross-validation**: always StratifiedGroupKFold grouped by subject (prevents data leakage)
- **Metrics**: balanced accuracy and macro F1 score
- **Sensor ablations**: all → feet (LF+RF) → individual sensors
- **Sampling rate**: 100 Hz (from `metadata['freq']`, defaults to 100 if missing)

## Data Flow

1. `load_trial()` reads raw sensor files + processed data + metadata JSON
2. **Traditional ML path**: `compute_trial_features()` extracts static features per trial → `train_baseline.py` runs CV
3. **Deep learning path**: `windowing.py` extracts raw time-series windows → `train_mamba_windows.py` trains TinyMambaTS
4. Both paths output to `results/artifacts/` and `results/figures/`

## Style

- Python PEP 8, 4-space indent, line length ≤ 88
- `snake_case` for functions/modules, `CapWords` for classes
- Type hints and docstrings for new functions
- Commits: Conventional Commits style (`feat(scope): ...`, `fix(scope): ...`)
