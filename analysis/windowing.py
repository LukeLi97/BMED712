import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.quick_start.load_data import load_trial  # type: ignore


def get_phase_bounds(md: dict, data_len: int) -> Dict[str, Tuple[int, int]]:
    """Return phase boundaries (inclusive start, exclusive end) in sample indices.

    Phases: pre_uturn, uturn, post_uturn, gait_full.
    """
    # uturn boundaries from metadata
    ub = md.get("uturnBoundaries") or [0, data_len - 1]
    us = int(max(0, min(ub[0], data_len - 1)))
    ue = int(max(us + 1, min(ub[1] if len(ub) > 1 else data_len - 1, data_len - 1)))

    # gait start/end from gait events if available
    left = md.get("leftGaitEvents") or []
    right = md.get("rightGaitEvents") or []
    if left and right:
        gs = int(max(0, min(left[0][0], right[0][0])))
        ge = int(min(data_len - 1, max(left[-1][1], right[-1][1])))
    else:
        gs, ge = 0, data_len - 1

    pre = (gs, max(gs, min(us, ge)))
    ut = (max(gs, min(us, ge)), max(us + 1, min(ue, ge + 1)))
    post = (max(ut[1], gs), ge + 1)
    full = (gs, ge + 1)
    return {
        "pre_uturn": pre,
        "uturn": ut,
        "post_uturn": post,
        "gait_full": full,
    }


def iter_windows(start: int, end: int, win: int, step: int) -> Iterable[Tuple[int, int]]:
    i = start
    while i + win <= end:
        yield i, i + win
        i += step


def spectral_features(x: np.ndarray, fs: float) -> Tuple[float, float, float]:
    if x.size <= 1 or fs <= 0:
        return 0.0, 0.0, 0.0
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x) if np.isfinite(np.nanmean(x)) else 0.0
    x0 = np.nan_to_num(x - m, copy=False)
    X = np.fft.rfft(x0)
    P = (X.real ** 2 + X.imag ** 2)
    freqs = np.fft.rfftfreq(x0.size, d=1.0 / fs)
    ps = float(np.sum(P))
    if P.size == 0 or ps <= 0:
        return 0.0, 0.0, 0.0
    di = int(np.argmax(P))
    dom = float(freqs[di]) if 0 <= di < freqs.size else 0.0
    sc = float(np.sum(freqs * P) / ps)
    return dom, sc, ps


def compute_window_features(df: pd.DataFrame, fs: float, i0: int, i1: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for c in df.columns:
        if c == "PacketCounter":
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        x = df[c].iloc[i0:i1].to_numpy(dtype=float)
        if x.size == 0:
            continue
        out[f"{c}__mean"] = float(np.nanmean(x))
        out[f"{c}__std"] = float(np.nanstd(x))
        out[f"{c}__rms"] = float(np.sqrt(np.nanmean(x ** 2)))
        d, sc, p = spectral_features(x, fs)
        out[f"{c}__dom_freq_hz"] = d
        out[f"{c}__spec_centroid_hz"] = sc
        out[f"{c}__spec_power"] = p
    out["duration_s"] = float(i1 - i0) / float(fs)
    return out


def build_windowed_table(
    base_path: str,
    trials: List[str],
    phase: str,
    win_s: float,
    overlap: float = 0.5,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    base = Path(base_path)
    for tr in trials:
        try:
            t = load_trial(str(base), tr)
        except Exception:
            continue
        md = t["metadata"]
        df = t["data_processed"]
        fs = float(md.get("freq", 100))
        bounds = get_phase_bounds(md, len(df))
        if phase not in bounds:
            continue
        s0, s1 = bounds[phase]
        win = max(2, int(round(win_s * fs)))
        step = max(1, int(round(win * (1.0 - overlap))))
        for i0, i1 in iter_windows(s0, s1, win, step):
            feats = compute_window_features(df, fs, i0, i1)
            feats["trial_id"] = tr
            feats["subject_id"] = str(md.get("subject", ""))
            feats["label"] = str(md.get("group", "unknown"))
            feats["phase"] = phase
            feats["win_s"] = float(win_s)
            feats["overlap"] = float(overlap)
            rows.append(feats)
    return pd.DataFrame(rows)


__all__ = [
    "build_windowed_table",
    "get_phase_bounds",
]

