"""English-oriented tokenizer whose boundary scan runs in Mojo."""

from __future__ import annotations

import re

import numpy as np

from ._lib import addr, lib
from .tokens import Doc

_SUFFIXES = ("'re", "'ve", "'ll", "'d", "'m", "'s", "’re", "’ve", "’ll", "’d", "’m", "’s")


def _split_contractions(text: str, start: int, end: int):
    word = text[start:end]
    if not any(char in word for char in ("'", "’")):
        return [(start, end)]
    remaining = word
    suffixes = []
    while True:
        lower = remaining.lower()
        suffix = next((value for value in _SUFFIXES if lower.endswith(value) and len(remaining) > len(value)), None)
        if suffix is None:
            break
        suffixes.append(len(suffix))
        remaining = remaining[: -len(suffix)]
    lower = remaining.lower()
    if lower.endswith("n't") and len(remaining) > 3:
        suffixes.append(3)
        remaining = remaining[:-3]
    elif lower.endswith("n’t") and len(remaining) > 3:
        suffixes.append(3)
        remaining = remaining[:-3]
    if not suffixes:
        return [(start, end)]
    spans = [(start, start + len(remaining))]
    cursor = spans[0][1]
    for length in reversed(suffixes):
        spans.append((cursor, cursor + length))
        cursor += length
    return spans


class Tokenizer:
    def __init__(
        self,
        vocab,
        rules=None,
        prefix_search=None,
        suffix_search=None,
        infix_finditer=None,
        token_match=None,
        url_match=None,
        faster_heuristics=True,
    ):
        self.vocab = vocab
        self.rules = rules or {}
        self.prefix_search = prefix_search
        self.suffix_search = suffix_search
        self.infix_finditer = infix_finditer
        self.token_match = token_match
        self.url_match = url_match
        self.faster_heuristics = faster_heuristics

    def __call__(self, text: str) -> Doc:
        if not isinstance(text, str):
            raise TypeError("Tokenizer input must be a string")
        if not text:
            return Doc(self.vocab, words=[], text="", indices=[], whitespace=[])
        chars = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
        starts = np.empty(len(chars) + 1, dtype=np.int64)
        ends = np.empty(len(chars) + 1, dtype=np.int64)
        kinds = np.empty(len(chars) + 1, dtype=np.uint8)
        count = lib().msp_tokenize(addr(chars), len(chars), addr(starts), addr(ends), addr(kinds))
        if not 0 <= count <= len(chars):
            raise RuntimeError(f"Mojo tokenizer returned invalid span count {count}")

        spans: list[list] = []
        for raw_start, raw_end, kind in zip(starts[:count], ends[:count], kinds[:count]):
            start, end = int(raw_start), int(raw_end)
            if kind:
                if start < end and text[start] == " " and spans and spans[-1][1] == start and not spans[-1][2]:
                    spans[-1][2] = " "
                    start += 1
                if start < end:
                    spans.append([start, end, ""])
                continue
            for split_start, split_end in _split_contractions(text, start, end):
                spans.append([split_start, split_end, ""])
        words = [text[start:end] for start, end, _ in spans]
        indices = [start for start, _, _ in spans]
        whitespace = [suffix for _, _, suffix in spans]
        return Doc(self.vocab, words=words, text=text, indices=indices, whitespace=whitespace)

    def pipe(self, texts, batch_size=1000):
        del batch_size
        for text in texts:
            yield self(text)

    def add_special_case(self, orth, case):
        self.rules[orth] = case

    def explain(self, text):
        return [("TOKEN", token.text) for token in self(text)]
