"""Feature extraction utilities for full-event and phase-specific segments.

This module builds on `dataset.quick_start.load_data` and computes fixed set
of global features per trial (one row per trial) suitable for traditional ML.

Guiding principles (see AGENTS.md):
- Do not alter dataset layout or filenames.
- Keep helpers under `dataset/quick_start/` with snake_case names.

Features per signal channel (e.g., `LF_FreeAcc_X`):
- mean, std, rms, ptp (peak-to-peak)
- domfreq (Hz): dominant frequency via FFT power spectrum
- bandpower: total spectral power (sum of power spectral density bins)
- entropy: spectral entropy (Shannon, normalized by log(N))

No external deps beyond numpy/pandas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Segment:
    start: int
    end: int

    def clamp(self, n_samples: int) -> "Segment":
        s = max(0, int(self.start))
        e = min(n_samples - 1, int(self.end))
        if e <= s:
            e = min(n_samples - 1, s + 1)
        return Segment(s, e)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def _ptp(x: np.ndarray) -> float:
    return float(np.ptp(x))


def _power_spectrum(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute one-sided power spectrum using FFT.

    Returns (freqs_hz, power)
    """
    n = len(x)
    if n <= 1:
        return np.array([0.0]), np.array([0.0])
    # Remove DC bias
    x = x - np.mean(x)
    # Hanning window to reduce leakage
    w = np.hanning(n)
    xw = x * w
    # FFT and one-sided spectrum
    fft = np.fft.rfft(xw)
    power = (np.abs(fft) ** 2) / np.sum(w**2)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, power


def _dominant_frequency(power_freqs: np.ndarray, power: np.ndarray) -> float:
    if power.size == 0:
        return 0.0
    idx = int(np.argmax(power))
    return float(power_freqs[idx])


def _bandpower(power: np.ndarray) -> float:
    return float(np.sum(power))


def _spectral_entropy(power: np.ndarray) -> float:
    if power.size == 0:
        return 0.0
    p = power.copy()
    s = np.sum(p)
    if s <= 0:
        return 0.0
    p /= s
    # Avoid log(0)
    p = np.clip(p, 1e-12, None)
    ent = -np.sum(p * np.log(p))
    # Normalize by log(N)
    ent /= np.log(len(p))
    return float(ent)


def compute_channel_features(series: pd.Series, fs: float, prefix: str) -> Dict[str, float]:
    """Compute features for a single channel.

    Names follow the pattern used in `master_features.csv`.
    """
    x = series.to_numpy(dtype=float)
    feats: Dict[str, float] = {}
    feats[f"{prefix}_mean"] = float(np.mean(x))
    feats[f"{prefix}_std"] = float(np.std(x, ddof=0))
    feats[f"{prefix}_rms"] = _rms(x)
    feats[f"{prefix}_ptp"] = _ptp(x)

    freqs, power = _power_spectrum(x, fs)
    feats[f"{prefix}_domfreq"] = _dominant_frequency(freqs, power)
    feats[f"{prefix}_bandpower"] = _bandpower(power)
    feats[f"{prefix}_entropy"] = _spectral_entropy(power)
    return feats


def select_segment_indices(
    df: pd.DataFrame, metadata: Dict, mode: str = "gait"
) -> Segment:
    """Return [start,end] indices for the chosen segment.

    mode:
      - "full": full recording
      - "gait": from first gait event to last (default)
      - "pre_uturn": from gait start to uturn start
      - "uturn": uturn boundaries
      - "post_uturn": from uturn end to gait end
    """
    n = len(df)
    if n == 0:
        return Segment(0, 0)

    # Derive gait start/end from events
    left = metadata.get("leftGaitEvents", [])
    right = metadata.get("rightGaitEvents", [])
    events = []
    for arr in (left, right):
        for pair in arr:
            if isinstance(pair, (list, tuple)) and len(pair) >= 1:
                events.extend(pair)
    if events:
        gait_start = int(np.min(events))
        gait_end = int(np.max(events))
    else:
        gait_start, gait_end = 0, n - 1

    uturn = metadata.get("uturnBoundaries", [gait_start, gait_end])
    u_start = int(uturn[0]) if len(uturn) > 0 else gait_start
    u_end = int(uturn[1]) if len(uturn) > 1 else gait_end

    if mode == "full":
        seg = Segment(0, n - 1)
    elif mode == "gait":
        seg = Segment(gait_start, gait_end)
    elif mode == "pre_uturn":
        seg = Segment(gait_start, u_start)
    elif mode == "uturn":
        seg = Segment(u_start, u_end)
    elif mode == "post_uturn":
        seg = Segment(u_end, gait_end)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return seg.clamp(n)


def compute_trial_features(
    df: pd.DataFrame,
    metadata: Dict,
    segment_mode: str = "gait",
    sensors: Iterable[str] = ("HE", "LB", "LF", "RF"),
    signal_kinds: Iterable[str] = ("FreeAcc", "Gyr"),
) -> Dict[str, float]:
    """Compute all features for a trial and return a flat dict.

    Only `FreeAcc_*` and `Gyr_*` channels are used to match reference CSV.
    """
    fs = float(metadata.get("freq", 100.0))
    seg = select_segment_indices(df, metadata, segment_mode)
    view = df.iloc[seg.start : seg.end + 1]

    out: Dict[str, float] = {}
    for sensor in sensors:
        for kind in signal_kinds:
            for axis in ("X", "Y", "Z"):
                col = f"{sensor}_{kind}_{axis}"
                if col in view.columns:
                    prefix = f"{sensor}_{kind}_{axis}"
                    out.update(compute_channel_features(view[col], fs, prefix))
    return out


def assemble_row(
    trial_id: str,
    subject_id: str,
    label: str,
    feature_dict: Dict[str, float],
) -> Dict[str, float | str]:
    row: Dict[str, float | str] = {**feature_dict}
    # Ensure stable tail columns consistent with reference
    row.update({"trial_id": trial_id, "subject_id": subject_id, "label": label})
    return row

