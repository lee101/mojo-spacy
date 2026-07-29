from __future__ import annotations

from .strings import StringStore
from .vectors import Vectors


class Vocab:
    def __init__(self, strings=(), *, vectors=None, lang="en"):
        self.strings = strings if isinstance(strings, StringStore) else StringStore(strings)
        self.vectors = vectors if vectors is not None else Vectors(self.strings, shape=(0, 0))
        self.lang = lang

    def __len__(self):
        return len(self.strings)

    def has_vector(self, key) -> bool:
        return key in self.vectors

    def get_vector(self, key):
        return self.vectors[key]

    def set_vector(self, key, vector):
        if self.vectors.shape[1] == 0:
            self.vectors.resize((max(100, self.vectors.shape[0]), len(vector)))
        elif self.vectors.is_full and key not in self.vectors:
            self.vectors.resize((max(1, self.vectors.shape[0] * 2), self.vectors.shape[1]))
        self.vectors.add(key, vector=vector)
