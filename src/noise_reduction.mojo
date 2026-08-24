"""Single-channel spectral subtraction and its C ABI.

All buffers are caller-owned float64 arrays. Complex spectra use NumPy's
interleaved complex128 layout.
"""

from std.math import cos, isfinite, pow, sin
from std.sys.info import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime PI = 3.141592653589793238462643383279502884


def is_power_of_two(n: Int) -> Bool:
    return n > 0 and (n & (n - 1)) == 0


def swap_complex(data: Ptr, a: Int, b: Int):
    var ar = data[2 * a]
    var ai = data[2 * a + 1]
    data[2 * a] = data[2 * b]
    data[2 * a + 1] = data[2 * b + 1]
    data[2 * b] = ar
    data[2 * b + 1] = ai


def fft_in_place(data: Ptr, n: Int, direction: Int):
    """Radix-2 complex FFT. direction=-1 is forward, +1 is inverse."""
    var j = 0
    for i in range(1, n):
        var bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            swap_complex(data, i, j)

    var length = 2
    while length <= n:
        var angle = Float64(direction) * 2.0 * PI / Float64(length)
        var step_r = cos(angle)
        var step_i = sin(angle)
        var half = length >> 1
        var base = 0
        while base < n:
            var wr = 1.0
            var wi = 0.0
            for k in range(half):
                var even = base + k
                var odd = even + half
                var ur = data[2 * even]
                var ui = data[2 * even + 1]
                var xr = data[2 * odd]
                var xi = data[2 * odd + 1]
                var vr = xr * wr - xi * wi
                var vi = xr * wi + xi * wr
                data[2 * even] = ur + vr
                data[2 * even + 1] = ui + vi
                data[2 * odd] = ur - vr
                data[2 * odd + 1] = ui - vi
                var next_wr = wr * step_r - wi * step_i
                wi = wr * step_i + wi * step_r
                wr = next_wr
            base += length
        length *= 2


def gain_filter_range(
    spectrum: Ptr,
    history: Ptr,
    signal_power: Ptr,
    noise_power: Ptr,
    gain: Ptr,
    n_bins: Int,
    history_length: Int,
    beta: Float64,
    alpha: Float64,
    gmin: Float64,
    start: Int,
    stop: Int,
):
    """Apply the minimum-statistics spectral-subtraction gain formula."""
    comptime W = simd_width_of[DType.float64]()
    var k = start
    if alpha == 1.0:
        var zeros = SIMD[DType.float64, W](0.0)
        var floors = SIMD[DType.float64, W](gmin)
        while k + W <= stop:
            var real_values = (spectrum + 2 * k).strided_load[width=W](2)
            var imag_values = (spectrum + 2 * k + 1).strided_load[width=W](2)
            var power_values = real_values * real_values + imag_values * imag_values
            signal_power.store(k, power_values)

            var row = k * history_length
            var minimum_values = power_values
            if history_length == 1:
                (history + row).strided_store[width=W](power_values, history_length)
            else:
                var oldest_values = (history + row).strided_load[width=W](
                    history_length
                )
                minimum_values = oldest_values
                for h in range(1, history_length - 1):
                    var values = (history + row + h).strided_load[width=W](
                        history_length
                    )
                    minimum_values = min(minimum_values, values)
                    (history + row + h - 1).strided_store[width=W](
                        values, history_length
                    )
                minimum_values = min(minimum_values, power_values)
                (history + row + history_length - 2).strided_store[width=W](
                    power_values, history_length
                )
                (history + row + history_length - 1).strided_store[width=W](
                    oldest_values, history_length
                )
            noise_power.store(k, minimum_values)
            var residual = max(power_values - beta * minimum_values, zeros)
            var ratios = power_values.gt(0.0).select(
                residual / power_values, zeros
            )
            gain.store(k, max(ratios, floors))
            k += W

    while k < stop:
        var re = spectrum[2 * k]
        var im = spectrum[2 * k + 1]
        var power_value = re * re + im * im
        signal_power[k] = power_value
        var row = k * history_length
        var oldest = history[row]
        history[row + history_length - 1] = power_value
        var minimum_values = SIMD[DType.float64, W](history[row])
        var h = 0
        if history_length > 1:
            while h + W <= history_length - 1:
                var values = history.load[width=W](row + h + 1)
                minimum_values = min(minimum_values, values)
                history.store(row + h, values)
                h += W
            while h < history_length - 1:
                var history_value = history[row + h + 1]
                minimum_values = min(minimum_values, history_value)
                history[row + h] = history_value
                h += 1
            history[row + history_length - 1] = oldest

        var minimum = minimum_values.reduce_min()
        noise_power[k] = minimum
        var value = gmin
        if power_value > 0.0:
            var residual = power_value - beta * minimum
            if residual < 0.0:
                residual = 0.0
            if alpha == 1.0:
                value = residual / power_value
            else:
                value = pow(residual / power_value, alpha)
            if value < gmin:
                value = gmin
        gain[k] = value
        k += 1


def gain_filter(
    spectrum: Ptr,
    history: Ptr,
    signal_power: Ptr,
    noise_power: Ptr,
    gain: Ptr,
    n_bins: Int,
    history_length: Int,
    beta: Float64,
    alpha: Float64,
    gmin: Float64,
):
    gain_filter_range(
        spectrum,
        history,
        signal_power,
        noise_power,
        gain,
        n_bins,
        history_length,
        beta,
        alpha,
        gmin,
        0,
        n_bins,
    )


def spectrum_is_finite(spectrum: Ptr, length: Int) -> Bool:
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= length:
        if not isfinite(spectrum.load[width=W](i)).reduce_and():
            return False
        i += W
    while i < length:
        if not isfinite(spectrum[i]):
            return False
        i += 1
    return True


def apply_spectral_sub_impl(
    source: Ptr,
    destination: Ptr,
    n: Int,
    nfft: Int,
    hop: Int,
    history_length: Int,
    beta: Float64,
    alpha: Float64,
    gmin: Float64,
    frame: Ptr,
    history: Ptr,
    norm: Ptr,
):
    var n_bins = nfft // 2 + 1
    var history_size = n_bins * history_length
    var window = history + history_size
    comptime W = simd_width_of[DType.float64]()
    var zeros = SIMD[DType.float64, W](0.0)
    var offset = 0
    while offset + W <= n:
        destination.store(offset, zeros)
        norm.store(offset, zeros)
        offset += W
    while offset < n:
        destination[offset] = 0.0
        norm[offset] = 0.0
        offset += 1

    var i = 0
    while i + W <= history_size:
        history.store(i, zeros)
        i += W
    while i < history_size:
        history[i] = 0.0
        i += 1
    for index in range(nfft):
        window[index] = 0.5 - 0.5 * cos(
            2.0 * PI * Float64(index) / Float64(nfft)
        )

    var frame_start = -(nfft >> 1)
    var history_slot = history_length - 1
    while frame_start < n:
        for i in range(nfft):
            var position = frame_start + i
            var sample = 0.0
            if position >= 0 and position < n:
                sample = source[position]
            frame[2 * i] = sample * window[i]
            frame[2 * i + 1] = 0.0

        fft_in_place(frame, nfft, -1)

        for k in range(n_bins):
            var re = frame[2 * k]
            var im = frame[2 * k + 1]
            var power_value = re * re + im * im
            var row = k * history_length
            history[row + history_slot] = power_value
            var minimum = history[row]
            for h in range(1, history_length):
                if history[row + h] < minimum:
                    minimum = history[row + h]

            var gain_value = gmin
            if power_value > 0.0:
                var residual = power_value - beta * minimum
                if residual < 0.0:
                    residual = 0.0
                if alpha == 1.0:
                    gain_value = residual / power_value
                else:
                    gain_value = pow(residual / power_value, alpha)
                if gain_value < gmin:
                    gain_value = gmin
            frame[2 * k] *= gain_value
            frame[2 * k + 1] *= gain_value

        for k in range(1, n_bins - 1):
            frame[2 * (nfft - k)] = frame[2 * k]
            frame[2 * (nfft - k) + 1] = -frame[2 * k + 1]

        fft_in_place(frame, nfft, 1)
        for i in range(nfft):
            var position = frame_start + i
            if position < 0 or position >= n:
                continue
            destination[position] += frame[2 * i] * window[i] / Float64(nfft)
            norm[position] += window[i] * window[i]

        history_slot += 1
        if history_slot == history_length:
            history_slot = 0
        frame_start += hop

    i = 0
    while i + W <= n:
        var norm_values = norm.load[width=W](i)
        var destination_values = destination.load[width=W](i)
        var normalized = norm_values.gt(1.0e-15).select(
            destination_values / norm_values, zeros
        )
        destination.store(i, normalized)
        i += W
    while i < n:
        if norm[i] > 1.0e-15:
            destination[i] /= norm[i]
        else:
            destination[i] = 0.0
        i += 1


@export("mnr_gain_filter_f64")
def mnr_gain_filter_f64(
    spectrum: Int,
    spectrum_length: Int,
    history: Int,
    history_capacity: Int,
    signal_power: Int,
    signal_power_length: Int,
    noise_power: Int,
    noise_power_length: Int,
    gain: Int,
    gain_length: Int,
    n_bins: Int,
    history_length: Int,
    beta: Float64,
    alpha: Float64,
    gmin: Float64,
) abi("C") -> Int:
    if (
        spectrum == 0
        or history == 0
        or signal_power == 0
        or noise_power == 0
        or gain == 0
        or n_bins <= 0
        or history_length <= 0
        or spectrum_length < 0
        or n_bins > spectrum_length // 2
        or history_capacity < 0
        or history_capacity // history_length < n_bins
        or signal_power_length < n_bins
        or noise_power_length < n_bins
        or gain_length < n_bins
        or not isfinite(beta)
        or not isfinite(alpha)
        or not isfinite(gmin)
        or beta < 0.0
        or alpha <= 0.0
        or gmin < 0.0
        or gmin > 1.0
    ):
        return 0
    var spectrum_pointer = Ptr(unsafe_from_address=spectrum)
    if not spectrum_is_finite(spectrum_pointer, 2 * n_bins):
        return -1
    gain_filter(
        spectrum_pointer,
        Ptr(unsafe_from_address=history),
        Ptr(unsafe_from_address=signal_power),
        Ptr(unsafe_from_address=noise_power),
        Ptr(unsafe_from_address=gain),
        n_bins,
        history_length,
        beta,
        alpha,
        gmin,
    )
    return 1


@export("mnr_apply_spectral_sub_f64")
def mnr_apply_spectral_sub_f64(
    source: Int,
    source_length: Int,
    destination: Int,
    destination_length: Int,
    n: Int,
    nfft: Int,
    hop: Int,
    history_length: Int,
    beta: Float64,
    alpha: Float64,
    gmin: Float64,
    frame: Int,
    frame_length: Int,
    history: Int,
    history_capacity: Int,
    norm: Int,
    norm_length: Int,
) abi("C") -> Int:
    if (
        source == 0
        or destination == 0
        or frame == 0
        or history == 0
        or norm == 0
        or n <= 0
        or nfft < 2
        or not is_power_of_two(nfft)
        or hop <= 0
        or hop > nfft
        or history_length <= 0
        or source_length < 0
        or destination_length < 0
        or frame_length < 0
        or history_capacity < 0
        or norm_length < 0
        or source_length < n
        or destination_length < n
        or norm_length < n
        or nfft > frame_length // 2
        or history_capacity < nfft
        or (history_capacity - nfft) // history_length < nfft // 2 + 1
        or not isfinite(beta)
        or not isfinite(alpha)
        or not isfinite(gmin)
        or beta < 0.0
        or alpha <= 0.0
        or gmin < 0.0
        or gmin > 1.0
    ):
        return 0
    apply_spectral_sub_impl(
        Ptr(unsafe_from_address=source),
        Ptr(unsafe_from_address=destination),
        n,
        nfft,
        hop,
        history_length,
        beta,
        alpha,
        gmin,
        Ptr(unsafe_from_address=frame),
        Ptr(unsafe_from_address=history),
        Ptr(unsafe_from_address=norm),
    )
    return 1
