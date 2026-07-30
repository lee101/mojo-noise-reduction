import inspect
import os

import numpy as np
import pytest

import noise_reduction as nr
from noise_reduction import _lib

from reference import ReferenceSpectralSub, apply_reference

rng = np.random.default_rng(20260730)


def test_public_api_and_signatures():
    assert nr.__all__ == [
        "SpectralSub",
        "apply_spectral_sub",
        "reduce_noise",
        "spectral_subtraction",
    ]
    expected = "(noisy_signal, nfft=512, db_reduc=25, lookback=12, beta=30, alpha=1)"
    assert str(inspect.signature(nr.apply_spectral_sub)) == expected
    assert str(inspect.signature(nr.reduce_noise)) == expected


@pytest.mark.parametrize("lookback", [0, 1, 5])
def test_gain_filter_matches_numpy_reference(lookback):
    ours = nr.SpectralSub(128, db_reduc=18, lookback=lookback, beta=2.5, alpha=1.3)
    reference = ReferenceSpectralSub(128, 18, lookback, 2.5, 1.3)
    for _ in range(12):
        spectrum = np.fft.rfft(rng.normal(size=128))
        np.testing.assert_allclose(
            ours.compute_gain_filter(spectrum),
            reference.compute_gain_filter(spectrum),
            rtol=2e-8,
            atol=2e-10,
        )
        np.testing.assert_allclose(ours.p_sn, reference.p_sn, rtol=2e-14, atol=2e-12)
        np.testing.assert_allclose(ours.p_n, reference.p_n, rtol=2e-14, atol=2e-12)
        np.testing.assert_allclose(ours.p_prev, reference.p_prev, rtol=2e-14, atol=2e-12)


def test_gain_filter_known_power_vector():
    ours = nr.SpectralSub(8, db_reduc=20, lookback=0, beta=2, alpha=1)
    spectrum = np.array([1, 2, 3, 4, 5], dtype=np.complex128)
    assert ours.compute_gain_filter(spectrum) == pytest.approx(np.full(5, 0.1))
    assert ours.p_sn == pytest.approx([1, 4, 9, 16, 25])
    assert ours.p_n == pytest.approx([1, 4, 9, 16, 25])


def test_gain_filter_zero_spectrum_is_finite():
    ours = nr.SpectralSub(16, db_reduc=30, lookback=0, beta=2)
    gain = ours.compute_gain_filter(np.zeros(9, dtype=np.complex128))
    assert np.all(np.isfinite(gain))
    assert gain == pytest.approx(np.full(9, ours.gmin))


def test_simd_tails_match_reference():
    ours = nr.SpectralSub(16, db_reduc=17, lookback=4, beta=2.25, alpha=1)
    reference = ReferenceSpectralSub(16, 17, 4, 2.25, 1)
    for _ in range(7):
        spectrum = np.fft.rfft(rng.normal(size=16))
        np.testing.assert_allclose(
            ours.compute_gain_filter(spectrum),
            reference.compute_gain_filter(spectrum),
            rtol=2e-8,
            atol=2e-10,
        )
        np.testing.assert_allclose(ours.p_prev, reference.p_prev, rtol=2e-14, atol=2e-12)

    source = rng.normal(size=259)
    np.testing.assert_allclose(
        nr.apply_spectral_sub(source, nfft=32, lookback=4, alpha=1),
        apply_reference(source, nfft=32, lookback=4, alpha=1),
        rtol=3e-8,
        atol=2e-9,
    )


def test_gain_filter_accepts_complex64_and_noncontiguous_input():
    spectrum = np.arange(18, dtype=np.float32).view(np.complex64)[::2]
    ours = nr.SpectralSub(8, 20, 1, 2)
    assert ours.compute_gain_filter(spectrum).shape == (5,)


@pytest.mark.parametrize(
    ("length", "nfft", "lookback", "beta", "alpha"),
    [
        (1, 32, 0, 1.0, 1.0),
        (31, 32, 1, 2.0, 1.0),
        (257, 64, 3, 1.5, 1.3),
        (2048, 128, 5, 4.0, 2.0),
        (16001, 512, 12, 30.0, 1.0),
    ],
)
def test_full_pipeline_matches_numpy_reference(length, nfft, lookback, beta, alpha):
    source = rng.normal(size=length)
    got = nr.apply_spectral_sub(
        source,
        nfft=nfft,
        db_reduc=22,
        lookback=lookback,
        beta=beta,
        alpha=alpha,
    )
    expected = apply_reference(source, nfft, 22, lookback, beta, alpha)
    np.testing.assert_allclose(got, expected, rtol=3e-8, atol=2e-9)


def test_defaults_match_numpy_reference():
    source = rng.normal(size=8000)
    np.testing.assert_allclose(
        nr.apply_spectral_sub(source),
        apply_reference(source),
        rtol=3e-8,
        atol=2e-9,
    )


def test_empty_input_roundtrips():
    got = nr.apply_spectral_sub([])
    assert got.dtype == np.float64
    assert got.shape == (0,)


def test_silence_remains_silence():
    assert np.array_equal(nr.apply_spectral_sub(np.zeros(4096)), np.zeros(4096))


def test_input_is_not_mutated_and_output_is_float64():
    source = rng.normal(size=1000).astype(np.float32)
    before = source.copy()
    got = nr.apply_spectral_sub(source, nfft=128)
    assert np.array_equal(source, before)
    assert got.dtype == np.float64
    assert got.shape == source.shape


def test_aliases_match_one_shot_function():
    source = rng.normal(size=2000)
    expected = nr.apply_spectral_sub(source, 128, 20, 3, 2, 1.5)
    assert np.array_equal(nr.reduce_noise(source, 128, 20, 3, 2, 1.5), expected)
    assert np.array_equal(
        nr.spectral_subtraction(source, 128, 20, 3, 2, 1.5), expected
    )


def test_stationary_noise_floor_is_reduced_after_warmup():
    source = rng.normal(scale=0.1, size=32000)
    denoised = nr.apply_spectral_sub(
        source, nfft=512, db_reduc=20, lookback=6, beta=10, alpha=1
    )
    warm = 4096
    assert np.sqrt(np.mean(denoised[warm:] ** 2)) < 0.75 * np.sqrt(
        np.mean(source[warm:] ** 2)
    )


@pytest.mark.parametrize("nfft", [0, 1, 3, 96, -8])
def test_invalid_fft_size_raises(nfft):
    with pytest.raises(ValueError, match="nfft"):
        nr.apply_spectral_sub(np.ones(10), nfft=nfft)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("lookback", -1),
        ("beta", -1),
        ("alpha", 0),
        ("db_reduc", -1),
        ("alpha", np.inf),
    ],
)
def test_invalid_algorithm_parameter_raises(keyword, value):
    with pytest.raises(ValueError):
        nr.apply_spectral_sub(np.ones(10), nfft=8, **{keyword: value})


def test_boolean_integer_parameters_are_rejected():
    with pytest.raises(TypeError):
        nr.apply_spectral_sub(np.ones(10), nfft=True)
    with pytest.raises(TypeError):
        nr.apply_spectral_sub(np.ones(10), nfft=8, lookback=False)


def test_multichannel_input_is_explicitly_rejected():
    with pytest.raises(ValueError, match="one-dimensional"):
        nr.apply_spectral_sub(np.ones((2, 100)))


def test_complex_signal_is_rejected_instead_of_narrowed():
    with pytest.raises(TypeError, match="real-valued"):
        nr.apply_spectral_sub(np.ones(10, dtype=np.complex128), nfft=8)


def test_nonfinite_input_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        nr.apply_spectral_sub([0.0, np.nan, 1.0], nfft=8)


def test_nonfinite_spectrum_is_rejected():
    subtractor = nr.SpectralSub(8, 20, 2, 3)
    spectrum = np.ones(5, dtype=np.complex128)
    spectrum[2] = np.inf + 1j
    with pytest.raises(ValueError, match="finite"):
        subtractor.compute_gain_filter(spectrum)


def test_wrong_spectrum_shape_is_rejected():
    subtractor = nr.SpectralSub(16, 20, 2, 3)
    with pytest.raises(ValueError, match="length 9"):
        subtractor.compute_gain_filter(np.ones(8, dtype=complex))
    with pytest.raises(ValueError, match="one-dimensional"):
        subtractor.compute_gain_filter(np.ones((1, 9), dtype=complex))


def test_shared_library_has_required_exports():
    path = _lib.build()
    assert os.path.basename(path) == "libmojo-noise-reduction.so"
    library = _lib.lib()
    assert library.mnr_gain_filter_f64
    assert library.mnr_apply_spectral_sub_f64


def test_ffi_rejects_null_and_undersized_buffers():
    library = _lib.lib()
    assert library.mnr_gain_filter_f64(
        0, 10, 0, 5, 0, 5, 0, 5, 0, 5, 5, 1, 2.0, 1.0, 0.1
    ) == 0
    assert library.mnr_gain_filter_f64(
        1, 10, 1, 5, 1, 5, 1, 5, 1, 5, 5, 1, np.nan, 1.0, 0.1
    ) == 0

    source = np.ones(8, dtype=np.float64)
    output = np.empty_like(source)
    frame = np.empty(16, dtype=np.float64)
    history = np.empty(13, dtype=np.float64)
    norm = np.empty_like(source)
    assert library.mnr_apply_spectral_sub_f64(
        source.ctypes.data, 7,
        output.ctypes.data, output.size,
        source.size, 8, 4, 1, 2.0, 1.0, 0.1,
        frame.ctypes.data, frame.size,
        history.ctypes.data, history.size,
        norm.ctypes.data, norm.size,
    ) == 0
