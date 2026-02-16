# BMED712 Track A — Phase-wise Single-Sensor Package (for discussion)

This folder contains the essential code, reports, and summary tables to discuss phase-wise (pre_u-turn / u-turn / post_u-turn / full gait) single-sensor results.

## Contents
- scripts/analysis/
  - compile_reports.py — builds Markdown report blocks and the full report.
  - export_report.py — converts Markdown to HTML (inlines figures).
  - phase_sensor_baselines.py — per-phase single-sensor 5-fold CV baselines.
  - plot_phase_single_vs_all.py — plots BAcc: single(best) vs all sensors by phase.
  - train_baseline.py, pipeline.py — broader baselines/utilities (optional).
- scripts/quick_start/
  - load_data.py, features.py — loaders and feature extraction per segment.
  - export_features.py — exports per-trial features for a chosen segment.
  - summarize_sensors.py — effect-size summary per phase × sensor.
- results/
  - report_full.html, report_full.md — primary report (open HTML in a browser).
  - report_onepage.md — one-page notes (if present).
  - sensor_phase_summary.csv — effect sizes + CV metrics for best sensor per phase.
  - sensor_phase_best.txt — quick best-per-phase sensor list.
  - figures/phase_single_vs_all.png — comparison chart (single vs all sensors).
  - artifacts/phase_single_vs_all.csv — table behind the chart.
- master_features.csv — reference for feature column ordering.
- requirements.txt — minimal Python packages.

## How to view
- Open `results/report_full.html` in a browser. It already inlines figures.
- Use the CSVs in `results/` for quick numeric inspection.

## How to regenerate (optional)
1) Create venv and install deps:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2) If dataset is available under `dataset/data/`, export features by phase:
   ```bash
   python -m scripts.quick_start.export_features --segment pre_uturn --template-csv master_features.csv --out results/features_pre_uturn.csv
   python -m scripts.quick_start.export_features --segment uturn --template-csv master_features.csv --out results/features_uturn.csv
   python -m scripts.quick_start.export_features --segment post_uturn --template-csv master_features.csv --out results/features_post_uturn.csv
   python -m scripts.quick_start.export_features --segment gait --template-csv master_features.csv --out results/features_gait_full.csv
   ```
3) Summarize single-sensor effect sizes & run baselines:
   ```bash
   python -m scripts.quick_start.summarize_sensors --out results/sensor_phase_summary.csv
   python scripts/analysis/phase_sensor_baselines.py --summary results/sensor_phase_summary.csv --out results/sensor_phase_summary.csv
   ```
4) Plot single vs all sensors by phase and rebuild the report:
   ```bash
   python scripts/analysis/plot_phase_single_vs_all.py
   python scripts/analysis/compile_reports.py
   python scripts/analysis/export_report.py
   ```

## Notes
- Dataset files are not included here to keep the package small; place the dataset under `dataset/data/` if you need to regenerate features.
- If PDF export fails (LaTeX/≈ symbol), use the HTML report and print to PDF from the browser.
