# Feature Extraction — Code Map and How-To

This note documents exactly how the per‑trial features were generated, with direct code references and a minimal reproduce script.

## What Features
- Per channel statistics: mean, std, rms, ptp.
- Frequency features from FFT: dominant frequency (Hz), total bandpower, spectral entropy (normalized).
- Channels included: sensors `HE`, `LB`, `LF`, `RF`; kinds `FreeAcc`, `Gyr`; axes `X`, `Y`, `Z`.
- Segment options: `gait` (default), `full`, `pre_uturn`, `uturn`, `post_uturn`.

## Exact Code
- Channel features: `BMED712 Project 1_Track A/dataset/quick_start/features.py:96`
  - `compute_channel_features(...)` computes mean/std/rms/ptp and FFT‑based features using `_power_spectrum`, `_dominant_frequency`, `_bandpower`, `_spectral_entropy`.
- Segment selection: `BMED712 Project 1_Track A/dataset/quick_start/features.py:115`
  - `select_segment_indices(...)` derives indices from metadata: `leftGaitEvents`, `rightGaitEvents`, `uturnBoundaries`; falls back to full recording if missing.
- Trial feature aggregation: `BMED712 Project 1_Track A/dataset/quick_start/features.py:165`
  - `compute_trial_features(...)` slices the chosen segment and aggregates features for all sensor×kind×axis channels; sampling rate from `metadata['freq']` (default 100 Hz).
- Row assembly: `BMED712 Project 1_Track A/dataset/quick_start/features.py:191`
  - `assemble_row(...)` appends `trial_id`, `subject_id`, `label` to the feature dict.
- Batch export CLI: `BMED712 Project 1_Track A/dataset/quick_start/export_features.py:75`
  - `export_features(...)` walks `dataset/data/<group>/<cohort>/<subject>/<trial>/`, loads `<trial>_processed_data.txt` + `<trial>_meta.json`, calls `compute_trial_features`, and writes a CSV aligned to `master_features.csv` if provided.
  - Entrypoint with arguments: `BMED712 Project 1_Track A/dataset/quick_start/export_features.py:117`.

Reference column template: `BMED712 Project 1_Track A/master_features.csv:1`.

## Reproduce Locally
From the repo root `BMED712 Project 1_Track A/`:

```bash
# 1) (optional) create a lightweight env
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas

# 2) Export features for the full‑gait segment
python -m dataset.quick_start.export_features \
  --data-root dataset/data \
  --segment gait \
  --template-csv master_features.csv \
  --out results/features_gait_full.csv

# Examples for other segments
python -m dataset.quick_start.export_features --data-root dataset/data --segment pre_uturn  --template-csv master_features.csv --out results/features_pre_uturn.csv
python -m dataset.quick_start.export_features --data-root dataset/data --segment uturn      --template-csv master_features.csv --out results/features_uturn.csv
python -m dataset.quick_start.export_features --data-root dataset/data --segment post_uturn --template-csv master_features.csv --out results/features_post_uturn.csv
```

Notes
- Data layout expected by the exporter: `dataset/data/<healthy|ortho|neuro>/<cohort>/<subject>/<trial>/` containing `*_processed_data.txt` (tab‑separated) and `*_meta.json`.
- If metadata lacks gait/uturn markers, the code uses the entire recording as the segment.
- Sampling rate defaults to 100 Hz via `metadata['freq']` if not present.
- The code under `share/for_professor_*` contains a frozen copy for packaging; the canonical implementation lives in `dataset/quick_start/` referenced above.

## One‑Line Summary
Features are computed in `dataset/quick_start/features.py` and exported in batch by `dataset/quick_start/export_features.py`; run the module with `--segment` to choose the phase and write CSVs aligned to `master_features.csv`.

