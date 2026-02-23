from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.windowing import get_phase_bounds, iter_windows  # type: ignore
from dataset.quick_start.load_data import load_data_processed, load_metadata  # type: ignore


def _list_trials(base: Path) -> List[str]:
    out: List[str] = []
    for top in ["healthy", "ortho", "neuro"]:
        tpath = base / top
        if not tpath.exists():
            continue
        for cohort in sorted(p for p in tpath.iterdir() if p.is_dir()):
            for subj in sorted(p for p in cohort.iterdir() if p.is_dir()):
                for tr in sorted(p for p in subj.iterdir() if p.is_dir()):
                    out.append(tr.name)
    return out


def _list_trials_by_group(base: Path) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {"healthy": [], "ortho": [], "neuro": []}
    for top in ["healthy", "ortho", "neuro"]:
        tpath = base / top
        if not tpath.exists():
            continue
        for cohort in sorted(p for p in tpath.iterdir() if p.is_dir()):
            for subj in sorted(p for p in cohort.iterdir() if p.is_dir()):
                for tr in sorted(p for p in subj.iterdir() if p.is_dir()):
                    # load metadata to confirm group label exists
                    try:
                        md = load_metadata(str(tr))  # wrong path; fix below
                    except Exception:
                        md = None
                    groups.setdefault(top, []).append(tr.name)
    return groups


def _load_trial(base: Path, trial: str) -> Tuple[pd.DataFrame, Dict]:
    parts = trial.split("_")
    if len(parts) < 3:
        raise ValueError(f"Bad trial name: {trial}")
    patient = f"{parts[0]}_{parts[1]}"
    cohort = parts[0]
    for top in ["healthy", "ortho", "neuro"]:
        p = base / top / cohort / patient / trial
        if p.exists():
            md = load_metadata(str(p))
            df = load_data_processed(str(p))
            return df, md
    raise FileNotFoundError(f"Not found: {trial}")


@dataclass
class WindowSpec:
    phase: str
    win_s: float
    overlap: float


def build_raw_windows(
    base_path: str,
    phase: str,
    win_s: float,
    overlap: float,
    sensor: str = "RF",
    limit_trials: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Return (X, y, groups, trials_used).

    X: [N, T, C] float32 time series windows for the given sensor.
    y: [N] int labels encoded from metadata['group'].
    groups: [N] subject ids as strings (grouped in CV).
    """
    base = Path(base_path)
    # choose balanced across groups if limit provided
    if limit_trials:
        per = max(1, int(limit_trials) // 3)
        all_trials: List[str] = []
        for top in ["healthy", "ortho", "neuro"]:
            tpath = base / top
            if not tpath.exists():
                continue
            acc = []
            for cohort in sorted(p for p in tpath.iterdir() if p.is_dir()):
                for subj in sorted(p for p in cohort.iterdir() if p.is_dir()):
                    for tr in sorted(p for p in subj.iterdir() if p.is_dir()):
                        acc.append(tr.name)
                        if len(acc) >= per:
                            break
                    if len(acc) >= per:
                        break
                if len(acc) >= per:
                    break
            all_trials.extend(acc)
    else:
        all_trials = _list_trials(base)
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    g_list: List[str] = []
    trials_used: List[str] = []
    label_map: Dict[str, int] = {}
    for tr in all_trials:
        try:
            df, md = _load_trial(base, tr)
        except Exception:
            continue
        fs = float(md.get("freq", 100))
        bounds = get_phase_bounds(md, len(df))
        if phase not in bounds:
            continue
        s0, s1 = bounds[phase]
        win = max(2, int(round(win_s * fs)))
        step = max(1, int(round(win * (1.0 - overlap))))
        # pick sensor columns
        cols = [c for c in df.columns if isinstance(c, str) and c.startswith(sensor + "_")]
        cols = [c for c in cols if c != "PacketCounter" and np.issubdtype(df[c].dtype, np.number)]
        if not cols:
            continue
        arr = df[cols].to_numpy(dtype=np.float32)
        for i0, i1 in iter_windows(s0, s1, win, step):
            seg = arr[i0:i1]
            if seg.shape[0] != win:
                continue
            X_list.append(seg)
            lab = str(md.get("group", "unknown"))
            if lab not in label_map:
                label_map[lab] = len(label_map)
            y_list.append(label_map[lab])
            g_list.append(str(md.get("subject", "")))
            trials_used.append(tr)
    if not X_list:
        return np.zeros((0, 1, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=object), []
    X = np.stack(X_list, axis=0)  # [N, T, C]
    y = np.asarray(y_list, dtype=np.int64)
    groups = np.asarray(g_list, dtype=object)
    return X, y, groups, trials_used


__all__ = ["build_raw_windows", "WindowSpec"]
