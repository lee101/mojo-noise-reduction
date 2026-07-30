import numpy as np


class ReferenceSpectralSub:
    def __init__(self, nfft, db_reduc, lookback, beta, alpha=1):
        self.beta = beta
        self.alpha = alpha
        self.n_bins = nfft // 2 + 1
        self.p_prev = np.zeros((self.n_bins, lookback + 1))
        self.gmin = 10 ** (-db_reduc / 20)
        self.p_sn = np.zeros(self.n_bins)
        self.p_n = np.zeros(self.n_bins)

    def compute_gain_filter(self, spectrum):
        self.p_sn[:] = spectrum.real**2 + spectrum.imag**2
        self.p_prev[:, -1] = self.p_sn
        self.p_n[:] = np.min(self.p_prev, axis=1)
        ratio = np.divide(
            np.maximum(self.p_sn - self.beta * self.p_n, 0),
            self.p_sn,
            out=np.zeros_like(self.p_sn),
            where=self.p_sn > 0,
        )
        gain = np.maximum(ratio**self.alpha, self.gmin)
        self.p_prev = np.roll(self.p_prev, -1, axis=1)
        return gain


def apply_reference(
    noisy_signal, nfft=512, db_reduc=25, lookback=12, beta=30, alpha=1
):
    source = np.asarray(noisy_signal, dtype=np.float64)
    destination = np.zeros_like(source)
    norm = np.zeros_like(source)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nfft) / nfft)
    subtractor = ReferenceSpectralSub(nfft, db_reduc, lookback, beta, alpha)
    hop = nfft // 2

    for start in range(-nfft // 2, source.size, hop):
        frame = np.zeros(nfft)
        lo = max(start, 0)
        hi = min(start + nfft, source.size)
        frame[lo - start : hi - start] = source[lo:hi]
        spectrum = np.fft.rfft(frame * window)
        gain = subtractor.compute_gain_filter(spectrum)
        reconstructed = np.fft.irfft(spectrum * gain, n=nfft)
        section = slice(lo - start, hi - start)
        destination[lo:hi] += reconstructed[section] * window[section]
        norm[lo:hi] += window[section] ** 2

    return np.divide(
        destination,
        norm,
        out=np.zeros_like(destination),
        where=norm > 1e-15,
    )
