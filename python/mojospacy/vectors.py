"""Dense float32 vectors with Mojo cosine and nearest-neighbor kernels."""

from __future__ import annotations

import numpy as np

from ._lib import addr, lib
from .strings import StringStore, hash_string

_COSINE_PARALLEL_THRESHOLD = 8_000_000
_COSINE_WORKERS = 4
_MOST_SIMILAR_PARALLEL_THRESHOLD = 8_000_000
_MOST_SIMILAR_WORKERS = 4


def _f32(value) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32:
        raise TypeError(
            f"vector buffers must use float32, got {array.dtype}; "
            "cast explicitly with np.asarray(value, dtype=np.float32)"
        )
    return np.ascontiguousarray(array)


def cosine_similarity(vector1, vector2) -> float:
    first = _f32(vector1).reshape(-1)
    second = _f32(vector2).reshape(-1)
    if first.shape != second.shape:
        raise ValueError(f"Shape mismatch: {first.shape} vs {second.shape}")
    if not first.size:
        return 0.0
    if first.size >= _COSINE_PARALLEL_THRESHOLD:
        scratch = np.empty(_COSINE_WORKERS * 16, dtype=np.float32)
        return lib().msp_cosine_parallel(
            addr(first), addr(second), first.size, addr(scratch), _COSINE_WORKERS
        )
    return lib().msp_cosine(addr(first), addr(second), first.size)


def normalize_vectors(vectors, *, inplace: bool = False) -> np.ndarray:
    if inplace:
        if not isinstance(vectors, np.ndarray):
            raise TypeError("inplace normalization requires a NumPy array")
        if vectors.dtype != np.float32 or not vectors.flags.c_contiguous or not vectors.flags.writeable:
            raise ValueError("inplace normalization requires a writable C-contiguous float32 array")
        result = vectors
    else:
        result = _f32(vectors).copy()
    if result.ndim == 0:
        raise ValueError("vectors must have at least one dimension")
    if not result.size:
        return result
    matrix = result.reshape(1, -1) if result.ndim == 1 else result.reshape(-1, result.shape[-1])
    lib().msp_normalize(addr(matrix), matrix.shape[0], matrix.shape[1])
    return result


class Vectors:
    def __init__(
        self,
        strings=None,
        *,
        shape=None,
        data=None,
        keys=None,
        name=None,
        mode="default",
        minn=0,
        maxn=0,
        hash_count=1,
        hash_seed=0,
        bow="<",
        eow=">",
        attr="ORTH",
    ):
        if mode not in ("default", None):
            raise NotImplementedError("floret mode is outside mojo-spacy's covered subset")
        if data is None:
            if shape is None:
                shape = (0, 0)
            self.data = np.zeros(tuple(shape), dtype=np.float32)
        else:
            self.data = _f32(data).copy()
            if self.data.ndim != 2:
                raise ValueError("data must be a two-dimensional array")
        self.strings = strings if strings is not None else StringStore()
        self.key2row: dict[int, int] = {}
        self.name = name
        self.attr = attr
        if keys is not None:
            for row, key in enumerate(keys):
                if row >= self.data.shape[0]:
                    break
                self.key2row[self._key(key)] = row

    def _key(self, key) -> int:
        if isinstance(key, str):
            if hasattr(self.strings, "add"):
                return int(self.strings.add(key))
            return hash_string(key)
        return int(key)

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def n_keys(self) -> int:
        return len(self.key2row)

    @property
    def is_full(self) -> bool:
        return len(set(self.key2row.values())) >= self.data.shape[0]

    def __len__(self):
        return self.data.shape[0]

    def __contains__(self, key):
        return self._key(key) in self.key2row

    def __getitem__(self, key):
        hashed = self._key(key)
        if hashed not in self.key2row:
            raise KeyError(key)
        return self.data[self.key2row[hashed]]

    def __setitem__(self, key, vector):
        hashed = self._key(key)
        if hashed not in self.key2row:
            self.add(hashed, vector=vector)
        else:
            self.data[self.key2row[hashed]] = _f32(vector)

    def keys(self):
        return self.key2row.keys()

    def values(self):
        return self.key2row.values()

    def items(self):
        return self.key2row.items()

    def add(self, key, *, vector=None, row=None):
        hashed = self._key(key)
        if row is None:
            if hashed in self.key2row:
                row = self.key2row[hashed]
            else:
                occupied = set(self.key2row.values())
                row = next((i for i in range(self.data.shape[0]) if i not in occupied), None)
                if row is None:
                    raise ValueError("The table is full. Resize it to add more vectors.")
        row = int(row)
        if not 0 <= row < self.data.shape[0]:
            raise ValueError(f"row {row} is outside vectors shape {self.shape}")
        if vector is not None:
            value = _f32(vector)
            if value.shape != (self.data.shape[1],):
                raise ValueError(f"vector must have shape {(self.data.shape[1],)}")
            self.data[row] = value
        self.key2row[hashed] = row
        return row

    def find(self, *, key=None, keys=None, row=None, rows=None):
        supplied = sum(value is not None for value in (key, keys, row, rows))
        if supplied != 1:
            raise ValueError("Exactly one of key, keys, row or rows is required")
        if key is not None:
            return self.key2row.get(self._key(key), -1)
        if keys is not None:
            return np.asarray([self.key2row.get(self._key(item), -1) for item in keys], dtype=np.int32)
        inverse = {}
        for item, item_row in self.key2row.items():
            inverse.setdefault(item_row, item)
        if row is not None:
            if int(row) not in inverse:
                raise KeyError(row)
            return inverse[int(row)]
        result = []
        for item_row in rows:
            if int(item_row) not in inverse:
                raise KeyError(item_row)
            result.append(inverse[int(item_row)])
        return np.asarray(result, dtype=np.uint64)

    def most_similar(self, queries, *, batch_size=1024, n=1, sort=True):
        del batch_size
        query_matrix = _f32(queries)
        if query_matrix.ndim == 1:
            query_matrix = query_matrix.reshape(1, -1)
        if query_matrix.ndim != 2 or query_matrix.shape[1] != self.data.shape[1]:
            raise ValueError(f"queries must have shape (n, {self.data.shape[1]})")
        valid_rows = np.asarray(sorted(set(self.key2row.values())), dtype=np.int64)
        if not len(valid_rows):
            raise ValueError("Vectors table has no assigned rows")
        if int(n) < 1:
            raise ValueError("n must be at least 1")
        n = min(int(n), len(valid_rows))
        if not len(query_matrix):
            empty_keys = np.empty((0, n), dtype=np.uint64)
            return empty_keys, empty_keys.astype(np.int32), empty_keys.astype(np.float32)
        best_rows64 = np.empty((len(query_matrix), n), dtype=np.int64)
        scores = np.empty((len(query_matrix), n), dtype=np.float32)
        arguments = (
            addr(self.data),
            addr(valid_rows),
            len(valid_rows),
            addr(query_matrix),
            len(query_matrix),
            self.data.shape[1],
            n,
            addr(best_rows64),
            addr(scores),
        )
        work = len(valid_rows) * len(query_matrix) * self.data.shape[1]
        if work >= _MOST_SIMILAR_PARALLEL_THRESHOLD and len(query_matrix) > 1:
            workers = min(_MOST_SIMILAR_WORKERS, len(query_matrix))
            lib().msp_most_similar_parallel(*arguments, workers)
        else:
            lib().msp_most_similar(*arguments)
        if not sort:
            pass
        inverse = {}
        for key, item_row in self.key2row.items():
            inverse.setdefault(item_row, key)
        best_keys = np.asarray(
            [[inverse[int(item_row)] for item_row in row] for row in best_rows64],
            dtype=np.uint64,
        )
        return best_keys, best_rows64.astype(np.int32), scores

    def resize(self, shape, inplace=False):
        del inplace
        rows, dims = map(int, shape)
        replacement = np.zeros((rows, dims), dtype=np.float32)
        copy_rows = min(rows, self.data.shape[0])
        copy_dims = min(dims, self.data.shape[1])
        replacement[:copy_rows, :copy_dims] = self.data[:copy_rows, :copy_dims]
        removed = [(key, row) for key, row in self.key2row.items() if row >= rows]
        self.key2row = {key: row for key, row in self.key2row.items() if row < rows}
        self.data = replacement
        return removed
