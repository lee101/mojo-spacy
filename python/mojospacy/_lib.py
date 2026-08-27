"""ctypes bindings for the single Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "kernels.mojo")
LIB = os.path.join(ROOT, "dist", "libmojo-spacy.so")

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "msp_tokenize": ([I, I, I, I, I], I),
    "msp_match": ([I] * 20, I),
    "msp_cosine": ([I, I, I], F),
    "msp_cosine_parallel": ([I, I, I, I, I], F),
    "msp_normalize": ([I, I, I], None),
    "msp_most_similar": ([I] * 9, None),
    "msp_most_similar_parallel": ([I] * 10, None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC):
        return LIB
    pixi = shutil.which("pixi")
    if shutil.which("mojo"):
        cmd = ["bash", os.path.join(ROOT, "build", "build.sh")]
    elif pixi:
        cmd = [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "build"]
    else:
        raise BuildError("Mojo compiler not found; install the Pixi environment first")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_LIB = None


def lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_LIB, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _LIB


def addr(array: np.ndarray) -> int:
    """Return an address only for arrays that are safe to expose to Mojo."""
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    address = int(array.ctypes.data)
    if array.size and address == 0:
        raise ValueError("FFI buffers must have a non-null data pointer")
    return address
