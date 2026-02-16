# Repository Guidelines

## Project Structure & Module Organization
- Data lives in `dataset/data/<cohort>/<subject>/<trial>/`.
- Quick-start utilities in `dataset/quick_start/`:
  - `load_data.py` (load cohorts/patients/trials)
  - `plot_data.py` (event/uturn visualization)
- Keep raw/processed/metadata filenames unchanged (e.g., `KOA_2_1_raw_data_LF.txt`, `*_processed_data.txt`, `*_meta.json`). Tools depend on these patterns.
- Add new helper modules under `dataset/quick_start/` without changing directory depth.
- Tests live in `tests/`.

## Build, Test, and Development Commands
- Create venv and install deps:
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `pip install pandas numpy matplotlib pytest`
- Quick sanity check (load a trial and inspect metadata):
  - `python -c "from dataset.quick_start.load_data import load_trial; t=load_trial('dataset/data','KOA_2_1'); print(t['metadata'].keys())"`
- Run tests: `pytest -q`
- Plot examples: call functions in `dataset/quick_start/plot_data.py` with the trial dict returned by `load_trial`.

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indentation, line length ≤ 88.
- Prefer type hints and docstrings for new functions.
- Naming: `snake_case` for functions/modules, `CapWords` for classes.
- Do not alter dataset filename patterns or directory depth.

## Testing Guidelines
- Framework: `pytest` with small sample trials.
- Assert expected keys/columns exist (e.g., `PacketCounter`, `*_Gyr_*`).
- Place tests in `tests/` and run with `pytest -q`.

## Commit & Pull Request Guidelines
- Commits: concise, imperative; use Conventional Commits when practical, e.g.:
  - `feat(quick_start): add trial loader helpers`
  - `fix(loader): handle missing LF sensor gracefully`
- PRs include: summary, motivation, scope; reproduction commands (and screenshots for plots); note any schema or naming changes (avoid when possible).

## Security & Data Handling
- Do not commit PHI or large archives. Keep minimal reproducible samples under `dataset/data` and reference external storage for full datasets.
- Avoid schema or naming changes; coordinate if unavoidable.

