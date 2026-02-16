#!/usr/bin/env bash
set -euo pipefail
python analysis/train_baseline.py
python analysis/compile_reports.py
python analysis/export_report.py
echo "Baseline regenerated: results/report_full.html"
