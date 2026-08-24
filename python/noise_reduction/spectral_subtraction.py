"""Single-channel noise reduction by minimum-statistics spectral subtraction."""

from __future__ import annotations

import math

import numpy as np

from ._lib import lib


def _integer(value, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result != value:
        raise ValueError(f"{name} must be an integer")
    if result < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _parameters(nfft, db_reduc, lookback, beta, alpha):
    nfft = _integer(nfft, "nfft", minimum=1)
    if nfft < 2 or nfft & (nfft - 1):
        raise ValueError("nfft must be a power of two and at least 2")
    lookback = _integer(lookback, "lookback", minimum=0)
    db_reduc = float(db_reduc)
    beta = float(beta)
    alpha = float(alpha)
    if not math.isfinite(db_reduc) or db_reduc < 0:
        raise ValueError("db_reduc must be finite and non-negative")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and non-negative")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    return nfft, db_reduc, lookback, beta, alpha


class SpectralSub:
    """Stateful spectral-subtraction gain estimator.

    Parameters match the established ``SpectralSub`` API. ``compute_gain_filter``
    accepts one complex RFFT spectrum and returns a float64 gain vector.
    """

    def __init__(self, nfft, db_reduc, lookback, beta, alpha=1):
        nfft, db_reduc, lookback, beta, alpha = _parameters(
            nfft, db_reduc, lookback, beta, alpha
        )
        self.beta = beta
        self.alpha = alpha
        self.n_bins = nfft // 2 + 1
        self.p_prev = np.zeros((self.n_bins, lookback + 1), dtype=np.float64)
        self.gmin = 10.0 ** (-db_reduc / 20.0)
        self.p_sn = np.zeros(self.n_bins, dtype=np.float64)
        self.p_n = np.zeros(self.n_bins, dtype=np.float64)
        self._gain = np.empty(self.n_bins, dtype=np.float64)

    def compute_gain_filter(self, X):
        spectrum = np.ascontiguousarray(X, dtype=np.complex128)
        if spectrum.ndim != 1 or spectrum.size != self.n_bins:
            raise ValueError(f"X must be a one-dimensional spectrum of length {self.n_bins}")
        ok = lib().mnr_gain_filter_f64(
            spectrum.ctypes.data,
            spectrum.size * 2,
            self.p_prev.ctypes.data,
            self.p_prev.size,
            self.p_sn.ctypes.data,
            self.p_sn.size,
            self.p_n.ctypes.data,
            self.p_n.size,
            self._gain.ctypes.data,
            self._gain.size,
            self.n_bins,
            self.p_prev.shape[1],
            self.beta,
            self.alpha,
            self.gmin,
        )
        if ok < 0:
            raise ValueError("X must contain only finite values")
        if not ok:
            raise RuntimeError("Mojo gain-filter kernel rejected valid inputs")
        return self._gain.copy()


def apply_spectral_sub(
    noisy_signal, nfft=512, db_reduc=25, lookback=12, beta=30, alpha=1
):
    """Denoise a one-dimensional real signal with spectral subtraction."""
    nfft, db_reduc, lookback, beta, alpha = _parameters(
        nfft, db_reduc, lookback, beta, alpha
    )
    if np.iscomplexobj(noisy_signal):
        raise TypeError("noisy_signal must be real-valued")
    source = np.ascontiguousarray(noisy_signal, dtype=np.float64)
    if source.ndim != 1:
        raise ValueError("noisy_signal must be one-dimensional")
    if source.size == 0:
        return np.empty(0, dtype=np.float64)
    if not np.all(np.isfinite(source)):
        raise ValueError("noisy_signal must contain only finite values")

    destination = np.empty_like(source)
    frame_size = 2 * nfft
    history_size = (nfft // 2 + 1) * (lookback + 1) + nfft
    scratch = np.empty(
        frame_size + history_size + source.size, dtype=np.float64
    )
    frame = scratch[:frame_size]
    history = scratch[frame_size : frame_size + history_size]
    norm = scratch[frame_size + history_size :]
    ok = lib().mnr_apply_spectral_sub_f64(
        source.ctypes.data,
        source.size,
        destination.ctypes.data,
        destination.size,
        source.size,
        nfft,
        nfft // 2,
        lookback + 1,
        beta,
        alpha,
        10.0 ** (-db_reduc / 20.0),
        frame.ctypes.data,
        frame.size,
        history.ctypes.data,
        history.size,
        norm.ctypes.data,
        norm.size,
    )
    if not ok:
        raise RuntimeError("Mojo spectral-subtraction kernel rejected valid inputs")
    return destination


def reduce_noise(
    noisy_signal, nfft=512, db_reduc=25, lookback=12, beta=30, alpha=1
):
    """Alias for :func:`apply_spectral_sub`."""
    return apply_spectral_sub(noisy_signal, nfft, db_reduc, lookback, beta, alpha)


def spectral_subtraction(
    noisy_signal, nfft=512, db_reduc=25, lookback=12, beta=30, alpha=1
):
    """Alias for :func:`apply_spectral_sub`."""
    return apply_spectral_sub(noisy_signal, nfft, db_reduc, lookback, beta, alpha)
