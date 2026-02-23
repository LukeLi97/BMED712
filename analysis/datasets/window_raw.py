import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.windowing import get_phase_bounds, iter_windows  # type: ignore
from dataset.quick_start.load_data import (  # type: ignore
    load_data_processed,
    load_metadata,
)


SENSOR_PREFIX = {
    "Acc": ["X", "Y", "Z"],
    "FreeAcc": ["X", "Y", "Z"],
    "Gyr": ["X", "Y", "Z"],
}


def _rf_channels_present(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for kind, axes in SENSOR_PREFIX.items():
        for ax in axes:
            name = f"RF_{kind}_{ax}"
            if name in df.columns:
                cols.append(name)
    return cols


def list_trials(base_path: Path) -> List[str]:
    trials: List[str] = []
    for top in ["healthy", "ortho", "neuro"]:
        top_path = base_path / top
        if not top_path.exists():
            continue
        for cohort in sorted(p for p in top_path.iterdir() if p.is_dir()):
            for subj in sorted(p for p in cohort.iterdir() if p.is_dir()):
                for tr in sorted(p for p in subj.iterdir() if p.is_dir()):
                    trials.append(tr.name)
    return trials


class WindowedRawIndex:
    """Index of (trial, i0, i1, label, subject) for raw RF windows."""

    def __init__(
        self,
        base_path: str,
        trials: Sequence[str],
        phase: str,
        win_s: float,
        overlap: float,
    ) -> None:
        self.base = Path(base_path)
        self.phase = phase
        self.win_s = float(win_s)
        self.overlap = float(overlap)
        self.entries: List[Tuple[str, int, int, str, str, float]] = []
        self.channels: List[str] = []
        self._trial_meta_cache: Dict[str, dict] = {}
        self._trial_df_cache: Dict[str, pd.DataFrame] = {}
        self._build(trials)

    def _load_trial(self, trial: str) -> Tuple[dict, pd.DataFrame]:
        if trial in self._trial_meta_cache and trial in self._trial_df_cache:
            return self._trial_meta_cache[trial], self._trial_df_cache[trial]
        # infer path: top/cohort/subject/trial
        parts = trial.split("_")
        if len(parts) < 3:
            raise FileNotFoundError(f"Bad trial name: {trial}")
        cohort = parts[0]
        subject = f"{parts[0]}_{parts[1]}"
        for top in ["healthy", "ortho", "neuro"]:
            tp = self.base / top / cohort / subject / trial
            if tp.exists():
                md = load_metadata(str(tp))
                df = load_data_processed(str(tp))
                self._trial_meta_cache[trial] = md
                self._trial_df_cache[trial] = df
                return md, df
        raise FileNotFoundError(f"Trial path not found for {trial}")

    def _build(self, trials: Sequence[str]) -> None:
        # discover canonical channel set from the first usable trial
        chan_ref: Optional[List[str]] = None
        for tr in trials:
            try:
                md, df = self._load_trial(tr)
            except Exception:
                continue
            cands = _rf_channels_present(df)
            if len(cands) >= 3:
                chan_ref = cands
                break
        self.channels = chan_ref or []

        for tr in trials:
            try:
                md, df = self._load_trial(tr)
            except Exception:
                continue
            fs = float(md.get("freq", 100))
            bounds = get_phase_bounds(md, len(df))
            if self.phase not in bounds:
                continue
            s0, s1 = bounds[self.phase]
            win = max(2, int(round(self.win_s * fs)))
            step = max(1, int(round(win * (1.0 - self.overlap))))
            # skip if not enough length
            if s1 - s0 < win:
                continue
            for i0, i1 in iter_windows(s0, s1, win, step):
                label = str(md.get("group", "unknown"))
                subject = str(md.get("subject", ""))
                self.entries.append((tr, i0, i1, label, subject, fs))

    def __len__(self) -> int:
        return len(self.entries)

    def labels(self) -> List[str]:
        return [e[3] for e in self.entries]

    def groups(self) -> List[str]:
        return [e[4] for e in self.entries]

    def get_window(self, idx: int) -> Tuple[np.ndarray, str, str]:
        tr, i0, i1, label, subject, fs = self.entries[idx]
        md, df = self._load_trial(tr)
        if not self.channels:
            chans = _rf_channels_present(df)
        else:
            chans = [c for c in self.channels if c in df.columns]
        X = np.stack([df[c].iloc[i0:i1].to_numpy(dtype=float) for c in chans], axis=0)
        return X, label, subject


class WindowedRawDataset:
    """PyTorch-style dataset wrapping WindowedRawIndex with fold-wise normalization."""

    def __init__(self, index: WindowedRawIndex):
        self.index = index
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    @property
    def channels(self) -> List[str]:
        return self.index.channels

    def __len__(self) -> int:
        return len(self.index)

    def compute_channel_stats(self, indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        c = len(self.channels)
        # Running mean/var (per-channel) over all time points
        sum_x = np.zeros((c,), dtype=np.float64)
        sum_x2 = np.zeros((c,), dtype=np.float64)
        n = 0
        for i in indices:
            x, _, _ = self.index.get_window(i)
            # x: [C, T]
            sum_x += np.nan_to_num(x, copy=False).sum(axis=1)
            sum_x2 += (np.nan_to_num(x, copy=False) ** 2).sum(axis=1)
            n += x.shape[1]
        mean = sum_x / max(1, n)
        var = sum_x2 / max(1, n) - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-8))
        self._mean, self._std = mean.astype(np.float32), std.astype(np.float32)
        return self._mean, self._std

    def set_norm(self, mean: np.ndarray, std: np.ndarray) -> None:
        self._mean = mean.astype(np.float32)
        self._std = std.astype(np.float32)

    def get(self, idx: int) -> Tuple[np.ndarray, str, str]:
        x, y, g = self.index.get_window(idx)
        if self._mean is not None and self._std is not None:
            x = (x - self._mean[:, None]) / (self._std[:, None] + 1e-8)
        else:
            # fallback: per-window z-score
            m = np.nanmean(x, axis=1, keepdims=True)
            s = np.nanstd(x, axis=1, keepdims=True) + 1e-8
            x = (x - m) / s
        return x.astype(np.float32), y, g

