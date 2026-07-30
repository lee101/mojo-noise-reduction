from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import noise_reduction as nr  # noqa: E402


class NumPySpectralSub:
    def __init__(self, nfft, db_reduc, lookback, beta, alpha=1):
        self.beta = beta
        self.alpha = alpha
        self.history = np.zeros((nfft // 2 + 1, lookback + 1))
        self.gmin = 10 ** (-db_reduc / 20)

    def compute_gain_filter(self, spectrum):
        power = spectrum.real**2 + spectrum.imag**2
        self.history[:, -1] = power
        noise = np.min(self.history, axis=1)
        ratio = np.divide(
            np.maximum(power - self.beta * noise, 0),
            power,
            out=np.zeros_like(power),
            where=power > 0,
        )
        gain = np.maximum(ratio**self.alpha, self.gmin)
        self.history = np.roll(self.history, -1, axis=1)
        return gain


def numpy_apply(source, nfft=512, db_reduc=25, lookback=12, beta=30, alpha=1):
    destination = np.zeros_like(source)
    norm = np.zeros_like(source)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nfft) / nfft)
    subtractor = NumPySpectralSub(nfft, db_reduc, lookback, beta, alpha)
    hop = nfft // 2
    for start in range(-nfft // 2, source.size, hop):
        frame = np.zeros(nfft)
        lo = max(start, 0)
        hi = min(start + nfft, source.size)
        section = slice(lo - start, hi - start)
        frame[section] = source[lo:hi]
        spectrum = np.fft.rfft(frame * window)
        restored = np.fft.irfft(
            spectrum * subtractor.compute_gain_filter(spectrum), n=nfft
        )
        destination[lo:hi] += restored[section] * window[section]
        norm[lo:hi] += window[section] ** 2
    return np.divide(destination, norm, out=destination, where=norm > 1e-15)


def timeit(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def main():
    rng = np.random.default_rng(0)
    spectra = np.fft.rfft(rng.normal(size=(2000, 512)), axis=1)
    audio_10s = rng.normal(size=160_000)
    audio_60s = rng.normal(size=960_000)

    def mojo_gains():
        subtractor = nr.SpectralSub(512, 25, 12, 30, 1)
        for spectrum in spectra:
            subtractor.compute_gain_filter(spectrum)

    def numpy_gains():
        subtractor = NumPySpectralSub(512, 25, 12, 30, 1)
        for spectrum in spectra:
            subtractor.compute_gain_filter(spectrum)

    cases = [
        ("gain filter, 2000 frames", mojo_gains, numpy_gains),
        (
            "one-shot, 10 s at 16 kHz",
            lambda: nr.apply_spectral_sub(audio_10s),
            lambda: numpy_apply(audio_10s),
        ),
        (
            "one-shot, 60 s at 16 kHz",
            lambda: nr.apply_spectral_sub(audio_60s),
            lambda: numpy_apply(audio_60s),
        ),
    ]

    nr.apply_spectral_sub(audio_10s)
    print(f"Machine: {platform.processor() or platform.machine()} ({platform.platform()})")
    print("| case | Mojo | NumPy reference | reference / Mojo |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_function, numpy_function in cases:
        mojo_time = timeit(mojo_function, repeat=3)
        numpy_time = timeit(numpy_function, repeat=3)
        ratio = numpy_time / mojo_time
        label = "faster" if ratio >= 1 else "slower"
        print(
            f"| {name} | {mojo_time * 1e3:.3f} ms | "
            f"{numpy_time * 1e3:.3f} ms | {ratio:.2f}x {label} |"
        )


if __name__ == "__main__":
    main()
