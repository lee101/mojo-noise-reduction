"""Build and load the Mojo spectral-subtraction library."""

from __future__ import annotations

import ctypes
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get(
    "MOJO_NOISE_REDUCTION_LIB",
    os.path.join(ROOT, "dist", "libmojo-noise-reduction.so"),
)
LIB_OVERRIDE = "MOJO_NOISE_REDUCTION_LIB" in os.environ
I = ctypes.c_int64
F = ctypes.c_double

_library: ctypes.CDLL | None = None


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "noise_reduction.mojo")
    stale = not os.path.exists(LIB) or os.path.getmtime(source) > os.path.getmtime(LIB)
    if force or stale:
        if LIB_OVERRIDE:
            raise BuildError(
                f"MOJO_NOISE_REDUCTION_LIB does not name an up-to-date library: {LIB}"
            )
        proc = subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode or not os.path.exists(LIB):
            raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        gain = _library.mnr_gain_filter_f64
        gain.argtypes = [I, I, I, I, I, I, I, I, I, I, I, I, F, F, F]
        gain.restype = I
        apply = _library.mnr_apply_spectral_sub_f64
        apply.argtypes = [I, I, I, I, I, I, I, I, F, F, F, I, I, I, I, I, I]
        apply.restype = I
    return _library
