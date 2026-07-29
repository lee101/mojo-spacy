"""spaCy-compatible string hashing and a compact bidirectional string store."""

from __future__ import annotations


def hash_string(value: str) -> int:
    data = value.encode("utf8")
    mask = (1 << 64) - 1
    multiplier = 0xC6A4A7935BD1E995
    result = (1 ^ (len(data) * multiplier)) & mask
    full = len(data) // 8 * 8
    for offset in range(0, full, 8):
        item = int.from_bytes(data[offset : offset + 8], "little")
        item = (item * multiplier) & mask
        item ^= item >> 47
        item = (item * multiplier) & mask
        result ^= item
        result = (result * multiplier) & mask
    for index, byte in enumerate(data[full:]):
        result ^= byte << (index * 8)
    if full != len(data):
        result = (result * multiplier) & mask
    result ^= result >> 47
    result = (result * multiplier) & mask
    result ^= result >> 47
    return result


class StringStore:
    def __init__(self, strings=()):
        self._strings: dict[int, str] = {}
        for string in strings:
            self.add(string)

    def add(self, string: str) -> int:
        key = hash_string(string)
        self._strings[key] = string
        return key

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.add(key)
        return self._strings[int(key)]

    def __contains__(self, key) -> bool:
        if isinstance(key, str):
            return hash_string(key) in self._strings
        return int(key) in self._strings

    def __iter__(self):
        return iter(self._strings.values())

    def __len__(self) -> int:
        return len(self._strings)
