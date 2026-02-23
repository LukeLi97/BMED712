from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def _rfft_power(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (freqs, power) for rFFT of zero-mean signal x.

    Power is magnitude-squared spectrum (no density scaling).
    """
    x = np.asarray(x, dtype=float)
    if x.size <= 1 or fs <= 0:
        return np.array([]), np.array([])
    x = np.nan_to_num(x - np.nanmean(x))
    X = np.fft.rfft(x)
    P = (X.real ** 2 + X.imag ** 2)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    return freqs, P


def rfft_freqs(n: int, fs: float) -> np.ndarray:
    if n <= 1 or fs <= 0:
        return np.array([])
    return np.fft.rfftfreq(n, d=1.0 / fs)


def band_power(x: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    """Compute simple rectangular band power between [f_lo, f_hi) Hz from rFFT.

    Uses magnitude-squared spectrum; adequate for relative comparisons across
    windows/trials when fs and window length are comparable.
    """
    x = np.asarray(x, dtype=float)
    if x.size <= 1 or fs <= 0 or f_hi <= f_lo:
        return 0.0
    x0 = np.nan_to_num(x - np.nanmean(x))
    X = np.fft.rfft(x0)
    P = (X.real ** 2 + X.imag ** 2)
    freqs = rfft_freqs(x0.size, fs)
    if freqs.size == 0:
        return 0.0
    idx = np.where((freqs >= f_lo) & (freqs < f_hi))[0]
    if idx.size == 0:
        return 0.0
    return float(np.sum(P[idx]))


def bandpowers_many(x: np.ndarray, fs: float, bands: Sequence[Tuple[float, float]]) -> List[float]:
    return [band_power(x, fs, a, b) for (a, b) in bands]


def add_bandpower_features(
    x: np.ndarray,
    fs: float,
    prefix: str,
    bands: Sequence[Tuple[float, float]],
) -> Dict[str, float]:
    """Return feature dict for provided bands using names like
    f"{prefix}__bp_{lo}_{hi}Hz" and pairwise ratios for interpretability.
    """
    vals = bandpowers_many(x, fs, bands)
    out: Dict[str, float] = {}
    for (lo, hi), v in zip(bands, vals):
        out[f"{prefix}__bp_{int(lo)}_{int(hi)}Hz"] = float(v)
    # add a few ratios (avoid division by zero)
    def safe_ratio(a: float, b: float) -> float:
        return float(a / b) if b not in (0.0, np.nan) else float("nan")

    if len(vals) >= 2:
        out[f"{prefix}__bp_ratio_{int(bands[0][0])}_{int(bands[0][1])}__{int(bands[1][0])}_{int(bands[1][1])}"] = safe_ratio(vals[0], vals[1])
    if len(vals) >= 3:
        out[f"{prefix}__bp_ratio_{int(bands[1][0])}_{int(bands[1][1])}__{int(bands[2][0])}_{int(bands[2][1])}"] = safe_ratio(vals[1], vals[2])
    return out


__all__ = [
    "band_power",
    "bandpowers_many",
    "add_bandpower_features",
    "rfft_freqs",
]


# -----------------------------
# Filterbank (frequency-windowing)
# -----------------------------

def _make_triangular_filterbank(freqs: np.ndarray, n_bands: int, fmax: float) -> np.ndarray:
    """Make a simple triangular filterbank up to fmax (inclusive).

    Returns weights of shape [n_bands, freqs.size]. Bands are linearly spaced in [0, fmax].
    """
    if n_bands <= 0 or freqs.size == 0:
        return np.zeros((0, freqs.size), dtype=float)
    fmax = float(max(1e-6, fmax))
    edges = np.linspace(0.0, fmax, num=n_bands + 2)
    W = np.zeros((n_bands, freqs.size), dtype=float)
    for b in range(n_bands):
        lo, ctr, hi = edges[b], edges[b + 1], edges[b + 2]
        # rising slope lo->ctr
        rising = (freqs >= lo) & (freqs <= ctr)
        falling = (freqs > ctr) & (freqs <= hi)
        if ctr > lo:
            W[b, rising] = (freqs[rising] - lo) / (ctr - lo)
        if hi > ctr:
            W[b, falling] = (hi - freqs[falling]) / (hi - ctr)
    # normalize each band so that sum of weights is 1
    s = W.sum(axis=1, keepdims=True)
    s[s == 0.0] = 1.0
    W = W / s
    return W


def add_filterbank_features(
    x: np.ndarray,
    fs: float,
    prefix: str,
    n_bands: int = 16,
    fmax: float = 20.0,
) -> Dict[str, float]:
    """Compute simple triangular filterbank energies on the rFFT power spectrum.

    This provides a fixed "frequency windowing" across bands for each window.
    """
    freqs, P = _rfft_power(x, fs)
    if P.size == 0:
        return {}
    W = _make_triangular_filterbank(freqs, int(n_bands), float(fmax))
    if W.size == 0:
        return {}
    energies = W @ P[: freqs.size]
    out: Dict[str, float] = {}
    for i, e in enumerate(energies, 1):
        out[f"{prefix}__fb_{i:02d}"] = float(e)
    # add a couple of ratios for robustness
    if energies.size >= 2:
        out[f"{prefix}__fb_ratio_01_02"] = float(energies[0] / energies[1]) if energies[1] != 0 else float("nan")
    if energies.size >= 3:
        out[f"{prefix}__fb_ratio_02_03"] = float(energies[1] / energies[2]) if energies[2] != 0 else float("nan")
    return out


__all__.extend(["add_filterbank_features"])
