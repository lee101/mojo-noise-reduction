# mojo-noise-reduction

`mojo-noise-reduction` ports the single-channel spectral-subtraction portion of
the upstream `noise-reduction` work to Mojo, exposed through a small
NumPy-compatible Python API.
The FFT, rolling noise estimate, gain calculation, and overlap-add reconstruction
run in one compiled Mojo call for the one-shot path.

This is deliberately not a port of the similarly named
[`noisereduce`](https://pypi.org/project/noisereduce/) spectral-gating package.
The API follows the established `SpectralSub` and `apply_spectral_sub` names and
signatures documented by
[pyroomacoustics](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.denoise.spectral_subtraction.html).
Tests compare it with an independent NumPy implementation of the same
minimum-statistics formula.

## Coverage

Covered:

- `SpectralSub(nfft, db_reduc, lookback, beta, alpha=1)`
- `SpectralSub.compute_gain_filter(X)` with compatible state attributes
  `p_prev`, `p_sn`, `p_n`, `gmin`, and `n_bins`
- `apply_spectral_sub(noisy_signal, nfft=512, db_reduc=25, lookback=12, beta=30, alpha=1)`
- convenience aliases `reduce_noise` and `spectral_subtraction`
- centered periodic-Hann STFT, minimum-over-lookback noise estimation, spectral
  flooring, and normalized overlap-add
- power-of-two FFT sizes and arbitrary one-dimensional signal lengths, including
  signals shorter than one frame

Not covered:

- multichannel input, arbitrary FFT sizes, GPU execution, streaming time-domain
  framing, audio file I/O, or resampling
- spectral gating, Wiener filtering, neural denoisers, or other algorithms from
  unrelated noise-reduction packages

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` writes `dist/libmojo-noise-reduction.so`. Importing the Python
package also rebuilds a missing or stale library. Set `MOJO_NOISE_REDUCTION_LIB`
to load a prebuilt library from another location.

## Usage

This complete example generates a noisy 440 Hz signal and denoises it:

```python
import numpy as np
from noise_reduction import apply_spectral_sub

sample_rate = 16_000
time = np.arange(2 * sample_rate) / sample_rate
rng = np.random.default_rng(0)
noisy = 0.4 * np.sin(2 * np.pi * 440 * time) + 0.08 * rng.normal(size=time.size)

denoised = apply_spectral_sub(
    noisy,
    nfft=512,
    db_reduc=20,
    lookback=12,
    beta=10,
    alpha=1,
)
print(denoised.shape, denoised.dtype)
```

For frame-oriented frequency-domain processing:

```python
from noise_reduction import SpectralSub

subtractor = SpectralSub(512, db_reduc=20, lookback=12, beta=10)
gain = subtractor.compute_gain_filter(np.fft.rfft(noisy[:512]))
```

## Correctness

The test suite compares every gain update and its rolling state against a separate
NumPy implementation. End-to-end tests compare the complete centered STFT and
overlap-add pipeline across short, partial, and long signals, multiple FFT sizes,
and varied suppression parameters. Behavioral tests cover silence, noise-floor
reduction, dtype conversion, input immutability, aliases, validation, and exported
library symbols.

Run the test suite with:

```bash
pixi run test
```

## Benchmarks

Measured by the final `pixi run bench` on this machine, a dual-socket Intel Xeon
E5-2697 v4 system running x86_64 Linux. Values are the best of three warm runs.
Both implementations use the same float64 algorithm and
parameters; the reference uses NumPy FFTs and array operations frame by frame.

| case | Mojo | NumPy reference | reference / Mojo |
| --- | ---: | ---: | ---: |
| gain filter, 2000 frames | 56.170 ms | 146.524 ms | 2.61x faster |
| one-shot, 10 s at 16 kHz | 16.415 ms | 82.636 ms | 5.03x faster |
| one-shot, 60 s at 16 kHz | 99.241 ms | 585.744 ms | 5.90x faster |

Reproduce the table under the repository-wide benchmark lock with:

```bash
pixi run bench
```

No GPU or parallel CPU path is included or benchmarked.

## How it works

Python owns the input, output, and scratch allocations and keeps every array alive
for the duration of each synchronous call. Contiguous float64 NumPy buffers cross
the C ABI as addresses accompanied by element capacities; Mojo rejects null or
undersized buffers before reconstructing pointers. Complex128 spectra use NumPy's
interleaved real/imaginary layout and pass their capacity in float64 elements.

The one-shot API crosses the FFI boundary once. Mojo centers each frame, applies a
cached periodic Hann window, runs an in-place radix-2 FFT, estimates each frequency
bin's noise power from a circular history buffer, applies the floored
spectral-subtraction gain, and reconstructs the signal with normalized overlap-add
at 50 percent hop. Native-width SIMD handles buffer initialization, final
normalization, and fused history minimum/shift updates with scalar remainder loops.
The common `alpha=1` case avoids a scalar power call. Scratch arrays share one
NumPy allocation, and `SpectralSub.compute_gain_filter` retains its rolling power
state in NumPy-owned memory with one small Mojo call per spectrum.

## License

MIT
