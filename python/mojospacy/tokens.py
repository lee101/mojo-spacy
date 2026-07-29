"""Small standalone implementations of spaCy's Doc, Token, and Span contracts."""

from __future__ import annotations

import math
import re
import unicodedata

import numpy as np

from .strings import hash_string
from .vectors import cosine_similarity
from .vocab import Vocab

_NUM_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion", "trillion",
}


def _shape(text: str) -> str:
    result = []
    run_value = None
    run_length = 0
    for char in text:
        if char.isupper():
            value = "X"
        elif char.islower():
            value = "x"
        elif char.isdigit():
            value = "d"
        else:
            value = char
        if value == run_value:
            run_length += 1
            if run_length <= 4:
                result.append(value)
        else:
            run_value = value
            run_length = 1
            result.append(value)
    return "".join(result)


def _is_punct(text: str) -> bool:
    return bool(text) and all(unicodedata.category(char).startswith("P") for char in text)


def _like_num(text: str) -> bool:
    lower = text.lower().replace(",", "")
    if lower in _NUM_WORDS:
        return True
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", lower):
        return True
    return bool(re.fullmatch(r"\d+/\d+", lower))


class Token:
    def __init__(self, doc: "Doc", i: int):
        self.doc = doc
        self.i = i

    @property
    def text(self):
        return self.doc._words[self.i]

    @property
    def text_with_ws(self):
        return self.text + self.whitespace_

    @property
    def whitespace_(self):
        return self.doc._whitespace[self.i]

    @property
    def idx(self):
        return self.doc._indices[self.i]

    @property
    def orth(self):
        return hash_string(self.text)

    @property
    def orth_(self):
        return self.text

    @property
    def lower(self):
        return hash_string(self.lower_)

    @property
    def lower_(self):
        return self.text.lower()

    @property
    def norm(self):
        return self.lower

    @property
    def norm_(self):
        return self.lower_

    @property
    def shape(self):
        return hash_string(self.shape_)

    @property
    def shape_(self):
        return _shape(self.text)

    @property
    def is_alpha(self):
        return self.text.isalpha()

    @property
    def is_digit(self):
        return self.text.isdigit()

    @property
    def is_ascii(self):
        return self.text.isascii()

    @property
    def is_space(self):
        return self.text.isspace()

    @property
    def is_punct(self):
        return _is_punct(self.text)

    @property
    def is_currency(self):
        return bool(self.text) and all(unicodedata.category(c) == "Sc" for c in self.text)

    @property
    def like_num(self):
        return _like_num(self.text)

    @property
    def is_stop(self):
        return False

    @property
    def has_vector(self):
        return self.text in self.doc.vocab.vectors

    @property
    def vector(self):
        if self.has_vector:
            return self.doc.vocab.vectors[self.text]
        return np.zeros(self.doc.vocab.vectors.shape[1], dtype=np.float32)

    @property
    def vector_norm(self):
        return float(np.linalg.norm(self.vector))

    def similarity(self, other):
        return cosine_similarity(self.vector, other.vector)

    def __len__(self):
        return len(self.text)

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text


class Span:
    def __init__(self, doc: "Doc", start: int, end: int, label=0, vector=None, vector_norm=None, kb_id=0, span_id=0):
        if start < 0 or end < start or end > len(doc):
            raise IndexError((start, end))
        self.doc = doc
        self.start = start
        self.end = end
        self.label = doc.vocab.strings.add(label) if isinstance(label, str) else int(label)
        self.kb_id = kb_id
        self.id = span_id
        self._vector = vector
        self._vector_norm = vector_norm

    @property
    def start_char(self):
        return self.doc._indices[self.start] if self.start < len(self.doc) else len(self.doc.text)

    @property
    def end_char(self):
        if self.end <= self.start:
            return self.start_char
        token = self.doc[self.end - 1]
        return token.idx + len(token)

    @property
    def text(self):
        return self.doc.text[self.start_char : self.end_char]

    @property
    def text_with_ws(self):
        if self.end <= self.start:
            return ""
        return self.doc.text[self.start_char : self.doc[self.end - 1].idx + len(self.doc[self.end - 1].text_with_ws)]

    @property
    def label_(self):
        if not self.label:
            return ""
        return self.doc.vocab.strings[self.label]

    @property
    def vector(self):
        if self._vector is not None:
            return self._vector
        if not len(self):
            return np.zeros(self.doc.vocab.vectors.shape[1], dtype=np.float32)
        return np.mean([token.vector for token in self], axis=0).astype(np.float32)

    @property
    def vector_norm(self):
        if self._vector_norm is not None:
            return self._vector_norm
        return float(np.linalg.norm(self.vector))

    def similarity(self, other):
        return cosine_similarity(self.vector, other.vector)

    def __len__(self):
        return self.end - self.start

    def __iter__(self):
        for i in range(self.start, self.end):
            yield self.doc[i]

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            if step != 1:
                return list(self)[item]
            return Span(self.doc, self.start + start, self.start + stop)
        if item < 0:
            item += len(self)
        return self.doc[self.start + item]

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text


class Doc:
    def __init__(self, vocab: Vocab, words=None, spaces=None, *, text=None, indices=None, whitespace=None):
        self.vocab = vocab
        if text is not None:
            self.text = text
            self._words = list(words or ())
            self._indices = list(indices or ())
            self._whitespace = list(whitespace or ("",) * len(self._words))
        else:
            self._words = list(words or ())
            if spaces is None:
                spaces = [True] * len(self._words)
                if spaces:
                    spaces[-1] = False
            if len(spaces) != len(self._words):
                raise ValueError("spaces must have the same length as words")
            self._whitespace = [" " if value else "" for value in spaces]
            self._indices = []
            cursor = 0
            pieces = []
            for word, suffix in zip(self._words, self._whitespace):
                self._indices.append(cursor)
                pieces.extend((word, suffix))
                cursor += len(word) + len(suffix)
            self.text = "".join(pieces)
        self.user_data = {}
        self._lexical_attrs = None

    @property
    def text_with_ws(self):
        return self.text

    def __len__(self):
        return len(self._words)

    def __iter__(self):
        for i in range(len(self)):
            yield Token(self, i)

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            if step != 1:
                return list(self)[item]
            return Span(self, start, stop)
        if item < 0:
            item += len(self)
        if not 0 <= item < len(self):
            raise IndexError(item)
        return Token(self, item)

    @property
    def has_vector(self):
        return any(token.has_vector for token in self)

    @property
    def vector(self):
        if not len(self):
            return np.zeros(self.vocab.vectors.shape[1], dtype=np.float32)
        return np.mean([token.vector for token in self], axis=0).astype(np.float32)

    @property
    def vector_norm(self):
        return math.sqrt(float(np.dot(self.vector, self.vector)))

    def similarity(self, other):
        return cosine_similarity(self.vector, other.vector)

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text
