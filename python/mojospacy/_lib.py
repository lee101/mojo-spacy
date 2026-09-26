"""ctypes bindings for the single Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

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
    "msp_normalize": ([I, I, I], None),
    "msp_most_similar": ([I] * 9, None),
    "msp_most_similar_range": ([I] * 10, None),
}

# A cosine reduction is three multiply-adds per element for eight bytes of
# input, under one flop per byte, so it stays serial. The nearest-neighbour
# search is different: every query re-reads the whole row matrix, and once that
# matrix is cache resident the dot-and-norm loop is compute bound. Mojo 1.2.0
# removed std.runtime.asyncrt, so the fan-out lives here and calls the
# range-taking export once per disjoint block of queries.
MOST_SIMILAR_PARALLEL_THRESHOLD = 8_000_000
MOST_SIMILAR_WORKERS = min(16, os.cpu_count() or 1)


def most_similar(
    data: np.ndarray,
    rows: np.ndarray,
    row_count: int,
    queries: np.ndarray,
    query_count: int,
    dims: int,
    nbest: int,
    best_rows: np.ndarray,
    scores: np.ndarray,
) -> None:
    """Fill ``best_rows`` and ``scores`` with the nearest neighbours per query."""
    arguments = (
        addr(data),
        addr(rows),
        row_count,
        addr(queries),
        query_count,
        dims,
        nbest,
        addr(best_rows),
        addr(scores),
    )
    if query_count < 2 or row_count * query_count * dims < MOST_SIMILAR_PARALLEL_THRESHOLD:
        lib().msp_most_similar(*arguments)
        return
    workers = min(MOST_SIMILAR_WORKERS, query_count)
    step = (query_count + workers - 1) // workers
    blocks = [(lo, min(lo + step, query_count)) for lo in range(0, query_count, step)]
    chunk = lib().msp_most_similar_range
    data_addr, rows_addr, queries_addr = arguments[0], arguments[1], arguments[3]
    best_addr, scores_addr = arguments[7], arguments[8]

    def run(bounds: tuple[int, int]) -> None:
        chunk(
            data_addr, rows_addr, row_count, queries_addr, bounds[0], bounds[1],
            dims, nbest, best_addr, scores_addr,
        )

    with ThreadPoolExecutor(max_workers=len(blocks)) as pool:
        list(pool.map(run, blocks))


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
